import sys
import json

def list_servers():
    servers = json.load(sys.stdin)['Reservations'][0]['Instances']

    print("Server                 Type                 Status")
    print("---------------------+--------------------+-------------")
    for server in servers:
        print(server["InstanceId"], "  ", f"{server['InstanceType']:20}", server.get("State",{}).get("Name", ""))


# Take first parameter and use it to call a function
if __name__ == '__main__':

    # args: [0] = current file, [1] = function name, [2:] = function args : (*unpacked)
    args = sys.argv
    globals()[args[1]](*args[2:])