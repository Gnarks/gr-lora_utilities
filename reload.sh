#!/bin/sh
rm -rf build/ && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make -j8 && sudo make install && sudo ldconfig && cd ..
