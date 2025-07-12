import numpy as np
from gnuradio import gr
from sigmf import SigMFFile
from sigmf.utils import get_sigmf_iso8601_datetime_now
import time
import socket
import threading
import os


class detectSyncFS(gr.sync_block):
    """
    docstring for block SyncFileSaver
    """

    def __init__(
        self,
        sample_rate=64000,
        distance=5,
        indoor=True,
        save_directory="~/Music/testSave/",
        scenario_name="poc",
        port=12345,
        ip_address="127.0.0.1",
        other_info="",
    ):
        gr.sync_block.__init__(
            self, name="Detection Sync FS", in_sig=[np.complex64], out_sig=None
        )

        # first, establish the TCP connection with the transmitter

        self.connected = False
        self.port = port
        self.ip_address = ip_address

        # set the current cycle at 0
        self.current_cycle = 0

        # root directory to save the scenario directory to
        self.save_directory = save_directory
        # name of directory to save the devices directories (where all the capture files belong)
        self.scenario_name = scenario_name

        # scenario context
        self.sample_rate = sample_rate
        self.description = (
            f"capture at {distance}m {'indoor' if indoor else 'outdoor'}. {other_info}"
        )
        self.wait_for_socket_connection()

    def work(self, input_items, output_items):
        print(
            f"from device {self.current_device} received {self.current_cycle} frame at {time.time()}"
        )
        device_received = self.current_device
        cycle_received = self.current_cycle

        # stop if all captured
        if self.current_cycle >= self.cycles:
            return 0

        # wait for the period to save the file
        self.save_frame_during_period(input_items, device_received, cycle_received)
        print(f"finished saving at {time.time()}")
        self.consume(0, len(input_items[0]))
        return 0

    def handle_socket(self):
        while True:
            try:
                input = self.get_next_var_from_sock(self.conn)
                # if nothing in the buffer
                if input == "":
                    continue

                if input == "close\n":
                    print("Closing connection")
                    self.current_cycle = self.cycles  # stop the receiving
                    self.conn.close()
                    os._exit(0)
                    return
                if input == "remove\n":
                    print(f"received remove at {time.time()}")

                    print("================================")
                    print(
                        f"the device {self.current_device} crashed ! removing it from the list"
                    )
                    print("================================")
                    # device id to be removed from the list
                    device_id = self.get_next_var_from_sock(self.conn)
                    restart_time = self.get_next_var_from_sock(self.conn)
                    restart_list_index = self.get_next_var_from_sock(self.conn)
                    self.handle_device_removal(
                        device_id, restart_time, restart_list_index
                    )

            except BlockingIOError:
                pass

    def handle_device_removal(self, device_id, restart_time, restart_list_index):
        # remove the crashed device to not cycle back to it
        self.device_list.remove(device_id)
        # set the next device to continue with
        self.current_index = restart_list_index
        self.current_device = self.device_list[self.current_index]

        # set a new start_time to loop on general work
        self.start_time = restart_time
        self.next = restart_time + self.period

    def handle_cycling_devices(self):
        """treaded function to keep track at any point in time from which device the frame is comming from"""
        while self.current_cycle < self.cycles:
            if time.time() < self.next:
                continue

            print(f"changed at {time.time()}")

            self.next += self.period
            self.next_device()

        # kill the thread if all cycles are done
        os._exit(0)

    def save_frame_during_period(self, data, device_received, cycle_received):
        """waiting until the next period"""

        self.save_frame(data, device_received, cycle_received)

        # create the metadata
        meta = SigMFFile(
            data_file=f"{self.save_directory}/{self.scenario_name}/device{device_received}/cycle_{cycle_received}.sigmf-data",  # extension is optional
            global_info={
                SigMFFile.DATATYPE_KEY: "cf32_le",
                SigMFFile.FREQUENCY_KEY: self.frequency,
                SigMFFile.SAMPLE_RATE_KEY: self.sample_rate,
                SigMFFile.DESCRIPTION_KEY: self.description,
                SigMFFile.DATETIME_KEY: get_sigmf_iso8601_datetime_now(),
            },
        )
        # save the meta file
        meta.tofile(
            f"{self.save_directory}/{self.scenario_name}/device{device_received}/cycle_{cycle_received}.sigmf-meta"
        )  # extension is optional

        print(f"{device_received} : [{cycle_received + 1}/{self.cycles}] saved")

    def save_frame(self, input_items, device_received, cycle_received):
        data = input_items[0]
        # create the data file and saves it
        filename = f"{self.save_directory}/{self.scenario_name}/device{device_received}/cycle_{cycle_received}.sigmf-data"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "ab") as f:
            np.array(data).tofile(f)

    def next_device(self):
        # change device
        print(f"moved from {self.current_device} ", end="")
        changeCycle = self.current_index + 1 == len(self.device_list)
        self.current_index = (self.current_index + 1) % len(self.device_list)
        self.current_device = self.device_list[self.current_index]

        print(f"to {self.current_device}")
        # if the last device in the list
        if changeCycle:
            self.current_cycle += 1
            print(f"next cycle :{self.current_cycle}")

        print()
        if self.current_cycle == self.cycles:
            print("all cycles are done !")

    def get_next_var_from_sock(self, sock):
        sep = "\n"
        data = sock.recv(1).decode("utf-8")
        buf = data
        # if deconnected
        if data == "":
            return ""

        # while not seen the separator
        while sep not in buf and data:
            buf += sock.recv(1).decode("utf-8")

        # the eval value is made for integers only
        if buf == "close\n" or buf == "remove\n":
            return buf

        if buf != "":
            data = eval(buf)
            return data

    def wait_for_socket_connection(self):
        print(f"Waiting for transmitter to connect to PORT:{self.port}")
        # by IP with TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.ip_address, self.port))

        sock.listen(5)
        self.conn, addr = sock.accept()
        print(f"Connection established with transmitter at {time.time()}")

        # transmission parameters
        self.frequency = self.get_next_var_from_sock(self.conn) * 1e6
        print(f"frequency received: {self.frequency}")

        self.sf = self.get_next_var_from_sock(self.conn)
        print(f"sf received: {self.sf}")

        self.bw = self.get_next_var_from_sock(self.conn) * 1e3
        print(f"bw received: {self.bw}")

        # device list to keep track of the origin of the signal
        self.device_list = self.get_next_var_from_sock(self.conn)
        print(f"device list received: {self.device_list}")

        # set the first device to be listened to
        self.current_device, self.current_index = self.device_list[0], 0

        # time synchronisation parameters
        self.start_time = self.get_next_var_from_sock(self.conn)
        print(
            f"start_time received: {self.start_time} compared to current {time.time()}"
        )

        self.period = self.get_next_var_from_sock(self.conn)  # should be given in s
        print(f"period received: {self.period}")

        self.cycles = self.get_next_var_from_sock(self.conn)
        print(f"cycles received: {self.cycles}")

        self.next = self.start_time + self.period

        self.connected = True
        # Start the thread to handle the device id
        self.socket_thread = threading.Thread(target=self.handle_socket)
        self.socket_thread.start()

        self.device_cycle_thread = threading.Thread(target=self.handle_cycling_devices)
        self.device_cycle_thread.start()
