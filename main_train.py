#!/usr/bin/env python

import pylearn2.utils.serial as serial

if __name__=='__main__':
  import sys
  assert(len(sys.argv) == 2)
  serial.load_train_file(sys.argv[1]).main_loop()
