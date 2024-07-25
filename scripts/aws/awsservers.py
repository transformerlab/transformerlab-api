import sys, json

servers = json.load(sys.stdin)['Reservations'][0]['Instances']

print("Server                 Type                 Status")
print("---------------------+--------------------+-------------")
for server in servers:
    print(server["InstanceId"], "  ", f"{server['InstanceType']:20}", server.get("State",{}).get("Name", ""))
