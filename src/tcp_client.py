'''
part of: cloud-image-segmentation
by: Daniel Casado Herraez

____________tcp_client.py____________
Initialize the TCP client for local inference
'''
from show import *
from infer_local import *
from config import *
from tcp import *

def start_client(args):
    display = False
    if args.display:
        display = True
        
    print("[CLIENT] Connecting to {} port {}".format(*client_connection))

    infer_local(args)


