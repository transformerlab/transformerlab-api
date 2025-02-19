# ##############################
# # Root Module Variables
# ##############################

# # The name of the resource group. This value is also passed into the resource_group and compute modules.
# resource_group_name = "rg-custom"           # e.g., "rg-myproject"

# # The Azure region where resources will be deployed.
# location = "eastus2"                        # e.g., "eastus", "westus2", etc.

# ##############################
# # Network Module Variables
# ##############################

# # Virtual Network configuration:
# vnet_name          = "vnet-custom"            # e.g., "vnet-myproject"
# vnet_address_space = ["10.1.0.0/16"]            # e.g., ["10.0.0.0/16"]

# # Subnet configuration:
# subnet_name             = "subnet-custom"       # e.g., "subnet-myproject"
# subnet_address_prefixes = ["10.1.1.0/24"]       # e.g., ["10.0.1.0/24"]

# # Public IP and Network Interface configuration:
# public_ip_name         = "pip-custom"           # e.g., "pip-myproject"
# network_interface_name = "nic-custom"           # e.g., "nic-myproject"

# # Network Security Group (NSG) configuration:
# nsg_name = "nsg-custom"                        # e.g., "nsg-myproject"

# # Define NSG security rules as a list of objects. You can add, remove, or modify rules.
# security_rules = [
#   {
#     name                       = "Allow-SSH"     # Rule name
#     priority                   = 1001            # Priority of the rule
#     direction                  = "Inbound"       # Direction of traffic
#     access                     = "Allow"         # Allow or Deny
#     protocol                   = "Tcp"           # Protocol (e.g., Tcp, Udp)
#     source_port_range          = "*"             # Source port (use "*" for any)
#     destination_port_range     = "22"            # Destination port
#     source_address_prefix      = "*"             # Source address (use "*" for any)
#     destination_address_prefix = "*"             # Destination address
#   },
#   {
#     name                       = "Allow-FastAPI"
#     priority                   = 1002
#     direction                  = "Inbound"
#     access                     = "Allow"
#     protocol                   = "Tcp"
#     source_port_range          = "*"
#     destination_port_range     = "8338"
#     source_address_prefix      = "*"
#     destination_address_prefix = "*"
#   },
#   {
#     name                       = "Allow-Electron"
#     priority                   = 1003
#     direction                  = "Inbound"
#     access                     = "Allow"
#     protocol                   = "Tcp"
#     source_port_range          = "*"
#     destination_port_range     = "1212"
#     source_address_prefix      = "*"
#     destination_address_prefix = "*"
#   }
# ]

# ##############################
# # Compute Module Variables
# ##############################

# # Virtual Machine configuration:
# vm_name   = "vm-custom"                      # e.g., "vm-myproject"
# vm_size   = "Standard_DS2_v2"                # e.g., "Standard_D8s_v3", "Standard_DS2_v2"
# admin_username = "adminuser"                 # e.g., "azureuser" or any admin name you prefer

# # OS Disk configuration:
# os_disk_storage_type = "Standard_LRS"         # e.g., "Premium_LRS", "StandardSSD_LRS"
# os_disk_size_gb      = 128                    # OS Disk size in GB

# # VM Image configuration:
# image_publisher = "Canonical"                 # e.g., "Canonical"
# image_offer     = "UbuntuServer"              # e.g., "0001-com-ubuntu-server-jammy"
# image_sku       = "18.04-LTS"                 # e.g., "22_04-lts-gen2" or "18.04-LTS"
# image_version   = "latest"                    # Use "latest" or a specific version number

# # Cloud-init configuration file that provisions the VM on boot.
# cloud_init_file = "cloud-init/cloud-init.yaml"  # Path relative to the root module

# # File path where the generated SSH private key will be saved.
# ssh_private_key_file = "~/.ssh/custom_key.pem"  # e.g., "~/.ssh/my_key.pem"
