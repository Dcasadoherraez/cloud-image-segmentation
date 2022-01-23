import sys

from show import *
from infer_local import *
from config import *
from tcp import *

def start_client(args):
    display = False
    if args.display:
        display = True
        
    print("[CLIENT] Connecting to {} port {}".format(*client_connection))

    cap = get_tcp_video()
    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    infer_local(args)


