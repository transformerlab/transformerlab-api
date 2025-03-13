#!/bin/bash

set -e

# Get the current Azure subscription ID via the Azure CLI and export it
subscription_id=$(az account show --query id -o tsv)

echo "Subscription ID: $subscription_id"

export ARM_SUBSCRIPTION_ID=${subscription_id}

# Function to extract a variable's value from terraform.tfvars (ignoring commented lines)
get_tfvar_value() {
  local var_name="$1"
  local default_value="$2"
  local line
  line=$(grep -E "^\s*${var_name}\s*=" terraform.tfvars | grep -v '^\s*#' || true)
  if [[ -n "$line" ]]; then
    value=$(echo "$line" | sed -E "s/.*=\s*\"([^\"]+)\".*/\1/")
  else
    value="$default_value"
  fi
  echo "$value"
}

# Function to check if a resource is already in the Terraform state.
is_imported() {
  local resource_name="$1"
  if terraform state list 2>/dev/null | grep -q "${resource_name}"; then
    return 0
  else
    return 1
  fi
}

# Get resource names from terraform.tfvars or use defaults
rg_name=$(get_tfvar_value "resource_group_name" "rg-transformerlab")
vnet_name=$(get_tfvar_value "vnet_name" "vnet-transformerlab")
subnet_name=$(get_tfvar_value "subnet_name" "subnet-transformerlab")
public_ip_name=$(get_tfvar_value "public_ip_name" "pip-transformerlab")
nic_name=$(get_tfvar_value "network_interface_name" "nic-transformerlab")
nsg_name=$(get_tfvar_value "nsg_name" "nsg-transformerlab")
vm_name=$(get_tfvar_value "vm_name" "vm-transformerlab")

echo "Using resource group: ${rg_name}"
echo "Using virtual network: ${vnet_name}"
echo "Using subnet: ${subnet_name}"
echo "Using public IP: ${public_ip_name}"
echo "Using network interface: ${nic_name}"
echo "Using NSG: ${nsg_name}"
echo "Using virtual machine: ${vm_name}"

# --- Resource Group ---
if is_imported "azurerm_resource_group.rg"; then
  echo "Resource group '${rg_name}' is already imported. Skipping import."
else
    if [[ $(az group exists --name "$rg_name") == "true" ]]; then
    echo "Resource group '${rg_name}' exists. Importing..."
    terraform import azurerm_resource_group.rg /subscriptions/${subscription_id}/resourceGroups/${rg_name} || true
    else
    echo "Resource group '${rg_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- Virtual Network ---
if is_imported "azurerm_virtual_network.vnet"; then
  echo "Virtual network '${vnet_name}' is already imported. Skipping import."
else
    if az network vnet show --resource-group "$rg_name" --name "$vnet_name" >/dev/null 2>&1; then
    echo "Virtual network '${vnet_name}' exists. Importing..."
    terraform import azurerm_virtual_network.vnet /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/virtualNetworks/${vnet_name} || true
    else
    echo "Virtual network '${vnet_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- Subnet ---
if is_imported "azurerm_subnet.subnet"; then
  echo "Subnet '${subnet_name}' is already imported. Skipping import."
else
    if az network vnet subnet show --resource-group "$rg_name" --vnet-name "$vnet_name" --name "$subnet_name" >/dev/null 2>&1; then
    echo "Subnet '${subnet_name}' exists. Importing..."
    terraform import azurerm_subnet.subnet /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/virtualNetworks/${vnet_name}/subnets/${subnet_name} || true
    else
    echo "Subnet '${subnet_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- Public IP ---
if is_imported "azurerm_public_ip.pip"; then
  echo "Public IP '${public_ip_name}' is already imported. Skipping import."
else
    if az network public-ip show --resource-group "$rg_name" --name "$public_ip_name" >/dev/null 2>&1; then
    echo "Public IP '${public_ip_name}' exists. Importing..."
    terraform import azurerm_public_ip.pip /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/publicIPAddresses/${public_ip_name} || true
    else
    echo "Public IP '${public_ip_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- Network Interface ---
if is_imported "azurerm_network_interface.nic"; then
  echo "Network interface '${nic_name}' is already imported. Skipping import."
else
    if az network nic show --resource-group "$rg_name" --name "$nic_name" >/dev/null 2>&1; then
    echo "Network interface '${nic_name}' exists. Importing..."
    terraform import azurerm_network_interface.nic /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/networkInterfaces/${nic_name} || true
    else
    echo "Network interface '${nic_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- Network Security Group ---
if is_imported "azurerm_network_security_group.nsg"; then
  echo "NSG '${nsg_name}' is already imported. Skipping import."
else
    if az network nsg show --resource-group "$rg_name" --name "$nsg_name" >/dev/null 2>&1; then
    echo "Network security group '${nsg_name}' exists. Importing..."
    terraform import azurerm_network_security_group.nsg /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/networkSecurityGroups/${nsg_name} || true
    else
    echo "Network security group '${nsg_name}' does not exist. It will be created by Terraform."
    fi
fi

# --- NSG association existence ---
# Note: There's no direct 'az' command to check the association. 
# Instead, you need to check if both the NIC and NSG exist and assume the association is in place.
if is_imported "azurerm_network_interface_security_group_association.nsg_assoc"; then
  echo "NSG association is already imported. Skipping import."
else
    if az network nic show --resource-group "$rg_name" --name "$nic_name" >/dev/null 2>&1 && \
    az network nsg show --resource-group "$rg_name" --name "$nsg_name" >/dev/null 2>&1; then
        echo "Network interface and NSG exist. Importing NSG association..."
        terraform import azurerm_network_interface_security_group_association.nsg_assoc "/subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/networkInterfaces/${nic_name}|/subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Network/networkSecurityGroups/${nsg_name}" || true
    else
        echo "Network interface or NSG does not exist. NSG association will be created by Terraform."
    fi
fi

# --- Linux Virtual Machine ---
if is_imported "azurerm_linux_virtual_machine.vm"; then
  echo "Virtual machine '${vm_name}' is already imported. Skipping import."
else
    if az vm show --resource-group "$rg_name" --name "$vm_name" >/dev/null 2>&1; then
    echo "Virtual machine '${vm_name}' exists. Importing..."
    terraform import azurerm_linux_virtual_machine.vm /subscriptions/${subscription_id}/resourceGroups/${rg_name}/providers/Microsoft.Compute/virtualMachines/${vm_name} || true
    else
    echo "Virtual machine '${vm_name}' does not exist. It will be created by Terraform."
    fi
fi

echo "Resource checks and imports completed."


# Provision all resources, VM and run startup script
terraform init -upgrade
terraform plan -out out.plan
terraform apply "out.plan"

# Save the SSH private key to a file
terraform output -raw ssh_private_key > ~/.ssh/az_vm_prvt_key.pem
chmod 600 ~/.ssh/az_vm_prvt_key.pem

## Connect to the VM via SSH
#ssh-keygen -R $(terraform output -raw public_ip)
#ssh -i ~/.ssh/az_vm_prvt_key.pem azureuser@$(terraform output -raw public_ip)

## Destroy the VM
#export ARM_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
#terraform destroy --target azurerm_virtual_machine.vm -auto-approve

## Destroy all resources
#terraform destroy -auto-approve

## Cleanup the SSH private key
#sshkeygen -R $(terraform output -raw public_ip)
#rm -f ~/.ssh/az_vm_prvt_key.pem