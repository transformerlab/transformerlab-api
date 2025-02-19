# Terraform & Azure Resource Deployment with Existing Resource Import

This folder provides a modular, production-grade Terraform setup for deploying Azure resources. It includes a helper script (`run.sh`) that automatically:

- Checks if the resources defined in your configuration already exist.
- Imports any existing resources into Terraform's state.
- Skips resources that are already managed by Terraform.
- Creates missing resources upon running `terraform apply`.

It also includes a cloud-init script to provision a Linux virtual machine that runs the Transformer Lab API service on Azure VM.

---

## Prerequisites

Before you begin, make sure you have the following installed:

- **Terraform**  
  [Install Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli) (v1.0+ recommended)

- **Azure CLI**  
  [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

- **Git** (optional, to clone this repository)

---

## Setup

1. **Azure Authentication**  
   Log in to your Azure account using the Azure CLI:
   ```bash
   az login

2. **Clone the Repository**
    If you haven't already, clone this repository:
    ```bash
    git clone https://github.com/transformerlab/transformerlab-api.git
    cd transformerlab-api/scripts/azure/terraform
    ```

3. **Configure Resource Names and Machine Types**
    The deployment uses a `terraform.tfvars` file to define resource names, locations, VM sizes, etc.

    - Edit `terraform.tfvars` (if needed) to set custom values.
    - Ensure that the variables you set (for example, `resource_group_name`, `vm_name`, etc.) are not commented out (i.e., remove any `#` at the beginning of lines).
    
    Example snippet from `terraform.tfvars`:

    ```bash
    # # The name of the resource group. This value is also passed into the resource_group and compute modules.
    # resource_group_name = "rg-custom"           # e.g., "rg-myproject"

    # # The Azure region where resources will be deployed.
    # location = "eastus2"                        # e.g., "eastus", "westus2", etc.
    ```

---

## Running the Deployment

1. **Run the Import/Creation Script**
    The provided `run.sh` script will check if each resource exists in Azure and if it is already imported into Terraform’s state. If a resource is found in Azure but not in the state, it will be imported automatically. If it does not exist, Terraform will create it on `terraform apply`.

    Make the script executable and run it:

    ```bash
    chmod +x run.sh
    ./run.sh

The script should take care of provisioning the resources and starting a VM with transformerlab api running inside. The script will also output the public IP of the VM on the command line to be used in the `Connect to Remote Engine` screen.

```bash
Apply complete! Resources: 1 added, 0 changed, 0 destroyed.

Outputs:

ssh_private_key = <sensitive>
subscription_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
vm_public_ip = "xxx.xxx.xxx.xxx"
```

---

## Post Deployment

1. **SSH into the VM**
    After deployment, you can SSH into the Linux VM using the generated private key. The key is saved (by default) in your home directory (e.g.,`~/.ssh/az_vm_prvt_key.pem`).

    Example SSH command:

    ```bash
    ssh-keygen -R <VM_PUBLIC_IP>
    ssh -i ~/.ssh/az_vm_prvt_key.pem azureuser@<VM_PUBLIC_IP>
    ```
    Replace `<VM_PUBLIC_IP>` with the public IP address output from Terraform.

2. **Viewing Transformer Lab Service Logs**
    The VM uses a systemd service named transformerlab.service to run the API. To check the service status or view logs for any errors:

    - Check the Service Status:

        ```bash
        sudo systemctl status transformerlab.service
        ```

    - View Service Logs:

        ```bash
        sudo journalctl -u transformerlab.service -f
        ```
    
    This command will stream live logs, allowing you to monitor the service in real time.

---

## Directory Structure

The directory structure for this repository is as follows:

```bash
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── terraform.tfvars      # User overrides for resource names and settings
├── run.sh                # Helper script to check and import resources
├── cloud-init/
│   └── cloud-init.yaml   # Cloud-init script for VM provisioning
└── modules/
    ├── resource_group/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── network/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── compute/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```