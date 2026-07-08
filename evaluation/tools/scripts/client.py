import socket

HOST = "127.0.0.1"
PORT = 5888

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[BBBB-SERVER] listens on {HOST}:{PORT}...")

    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)
            print(f"[BBBB-SERVER] Received from {addr}: {data.hex().upper()}")
