import socket
import time

HOST = "127.0.0.1"
PORT = 5888
MESSAGE = b"\xaa\xaa"

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(MESSAGE)
            print(f"[TESTCLIENT] Sent: {MESSAGE.hex().upper()}")
    except Exception as e:
        print(f"[TESTCLIENT] Error: {e}")
    time.sleep(1)
