import socket

ips = socket.getaddrinfo('images.cocodataset.org', 80)
print("IPs for images.cocodataset.org:")
for ip in set([item[4][0] for item in ips]):
    print(" ", ip)

