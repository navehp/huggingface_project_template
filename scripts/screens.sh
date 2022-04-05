START_AGENT=1
NUM_AGENTS=4
VENV_NAME=template

CREATE=true
#CREATE=false

ACTIVATE=true
#ACTIVATE=false

#CANCEL=true
CANCEL=false

#KILL=true
KILL=false

CONFIG_CUDA=true
#CONFIG_CUDA=false


for AGENT in $(seq $START_AGENT $NUM_AGENTS)
do
  SCREEN_NAME="agent${AGENT}"

  # Create New Screens
  if $CREATE;
  then
    screen -S ${SCREEN_NAME} -d -m bash
    echo "Created ${SCREEN_NAME}"
  fi

  # Activate Venv
  if $ACTIVATE;
  then
    screen -r ${SCREEN_NAME} -X stuff "conda activate ${VENV_NAME}\n"
    echo "Activated venv in ${SCREEN_NAME}"
  fi

  # Cancel Runs
  if $CANCEL;
  then
    screen -r ${SCREEN_NAME} -X stuff "^c\n"
    echo "Canceled runs in ${SCREEN_NAME}"
  fi

  # Kill Screen
  if $KILL;
  then
    screen -XS ${SCREEN_NAME} quit
    echo "Killed ${SCREEN_NAME}"
  fi
done

# Allocate GPUs
if $CONFIG_CUDA;
then
  screen -r "agent1" -X stuff "export CUDA_VISIBLE_DEVICES=0\n"
  echo "Configured Cuda in agent1"
  screen -r "agent2" -X stuff "export CUDA_VISIBLE_DEVICES=0\n"
  echo "Configured Cuda in agent2"
  screen -r "agent3" -X stuff "export CUDA_VISIBLE_DEVICES=1\n"
  echo "Configured Cuda in agent3"
  screen -r "agent4" -X stuff "export CUDA_VISIBLE_DEVICES=1\n"
  echo "Configured Cuda in agent4"
fi

