#!/bin/bash

# WARNING: THIS SCRIPT IS A WORK IN PROGRESS.
CONFIGURATION_FILE="aws.conf"

# Temporary variables that will be stored in config
AWS_ACCOUNT=""
AWS_SECURITY_GROUP=""
AWS_KEYNAME=""

# Constants
AWS_AMI_SEARCH_TEMPLATE="Deep Learning Base OSS Nvidia Driver AMI (Amazon Linux 2) Version ??.?"
AWS_SERVER_REGION="us-east-1"
AWS_SERVER_INSTANCETYPE="g5.xlarge"
AWS_SERVER_NAME="TransformerLabServer"
AWS_SERVER_VOLUMESIZE=200

##############################
# Helper Functions
##############################

err_report() {
  echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

abort() {
  printf "%s\n" "$@" >&2
  exit 1
}

###################################
# AWS Management and Setup
###################################

check_aws_cli_installed() {
    echo "Checking for AWS CLI..."
    AWS_CLI_CMD="which aws"
    if ! command -v aws &> /dev/null; then
        abort "❌ AWS CLI is not installed. Please install using these instructions: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        #echo "❌ AWS CLI is not installed. Please install using these instructions: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    fi

    # store AWS CLI version in variable:
    AWS_CLI_VERSION=$(aws --version)
    echo "✅ AWS CLI is installed: $AWS_CLI_VERSION"
}

# Call this to set necessary global variables
# Assumes setup has been run, otherwise will throw an error
aws_init() {
    check_aws_cli_installed

    # Read in configuration
    if [ ! -f $CONFIGURATION_FILE ]; then
        abort "❌ Failed to load configuration file $CONFIGURATION_FILE: File not found!"
    fi
    source $CONFIGURATION_FILE
    echo "✅ Loaded configuration from $CONFIGURATION_FILE"
    echo
}

aws_status() {
    aws_init
    echo "AWS Account: $AWS_ACCOUNT"
    echo "AWS Security Group: $AWS_SECURITY_GROUP"
    echo "AWS Key Name: $AWS_KEYNAME"

}

###################################
# Transformer Lab Server Management
###################################

aws_server_create() {
    echo "Building new AWS Server..."
    aws_init

    # Check for the latest AMI for Amazon Linux 2 with OSS NVIDIA driver
    AMI_CMD="aws ec2 describe-images --region $AWS_SERVER_REGION --owners amazon \
        --filters 'Name=name,Values=$AWS_AMI_SEARCH_TEMPLATE' 'Name=state,Values=available' \
        --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text"
    echo $AMI_CMD
    AMI_ID="1234567890"
    
    # Check for the latest AMI for Amazon Linux 2 with OSS NVIDIA driver
    CREATE_CMD="aws ec2 run-instances --count 1 --instance-type $AWS_SERVER_INSTANCETYPE \
        --key-name $AWS_KEYNAME \
        --security-group-ids $AWS_SECURITY_GROUP \
        --image-id $AMI_ID \
        --block-device-mapping DeviceName=/dev/xvda,Ebs={VolumeSize=$AWS_SERVER_VOLUMESIZE} \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$AWS_SERVER_NAME}]'"
    echo $CREATE_CMD
    AWS_INSTANCE_ID="i-1234567890"

    echo "Created server instance $AWS_INSTANCE_ID"
}

aws_server_start() {
    if [ -z "$1" ]
      then
        echo "Usage:"
        echo "  aws.sh server_start <servername>"
        abort "No server argument provided."
    fi

    echo "Starting AWS Server..."
    aws_init
    START_CMD="aws ec2 start-instances --instance-ids $1"
    echo $START_CMD
    echo "Server Started!"
}

aws_server_stop() {
    if [ -z "$1" ]
      then
        echo "Usage:"
        echo "  aws.sh server_stop <servername>"
        abort "No server argument provided."
    fi

    echo "Stopping AWS Server..."
    aws_init
    STOP_CMD="aws ec2 stop-instances --instance-ids $1"
    echo $STOP_CMD
    echo "Server Stopped!"
}

print_usage_message() {
  echo "usage: aws.sh <command>"
  echo ""
  echo "Commands:"
  echo "    create_server             Builds a new Transformer Lab server."
  echo "    start_server <instance>   Starts Transformer Lab server with id <instance> if it exists."
  echo "    stop_server <instance>    Stops Transformer Lab server with id <instance> if it exists."
  echo "    status                    Validates AWS setup and config and reports on status."
}

# Check if there are arguments to this script, and if so, run the appropriate function.
if [[ "$#" -eq 0 ]]; then
  print_usage_message
else
  case $1 in
    create_server)
      aws_server_create
      ;;
    start_server)
      aws_server_start "$2"
      ;;
    stop_server)
      aws_server_stop "$2"
      ;;
    status)
      aws_status
      ;;
    *)
      # Print allowed arguments
      echo "Allowed arguments: [setup, check_setup, create_server, start_server, stop_server]"
      abort "❌ Unknown argument: $arg"
      ;;
  esac
fi
