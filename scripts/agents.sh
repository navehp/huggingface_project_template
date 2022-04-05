SWEEP_ID=o4lxp1tt
START_AGENT=1
NUM_AGENTS=4

for AGENT in $(seq $START_AGENT ${NUM_AGENTS})
do
  SCREEN_NAME="agent${AGENT}"
  # Launch an agent
 screen -r ${SCREEN_NAME} -X stuff "wandb agent navehp/project_template/${SWEEP_ID}\n"
  echo "Launched agent in ${SCREEN_NAME}"
done