#!/bin/bash
ROS_DISTRO=humble

RECORD_EPISODES="$HOME/meloha_ws/src/meloha/scripts/record_episodes.py"
DUAL_ARM_TELEOP="$HOME/meloha_ws/src/meloha/scripts/dual_arm_teleop.py"
SLEEP="$HOME/meloha_ws/src/meloha/meloha/sleep.py"

print_usage() {
  echo "USAGE:"
  echo "auto_record.sh task num_episodes"
}

nargs="$#" # 스크립트에 전달된 인자의 개수

if [ $nargs -lt 2 ]; then # -lt == '<'
  echo "Passed incorrect number of arguments"
  print_usage
  exit 1
fi

if [ "$2" -lt 0 ]; then
  echo "# of episodes not valid"
  exit 1
fi

echo "Task: $1"
for (( i=0; i<$2; i++ ))
do
  echo "Starting episode $i"
  python3 "$RECORD_EPISODES" --task $1
  if [ $? -ne 0 ]; then
    echo "Failed to execute command. Returning"
    exit 1
  fi
  python3 "$DUAL_ARM_TELEOP"
  if [ $? -ne 0 ]; then
    echo "Failed to execute command. Returning"
    exit 1
  fi
  ros2 topic pub -1 /multi_set_position \
   multi_dynamixel_interfaces/msg/MultiSetPosition \
    '{ids: [0, 1, 3, 5, 6, 7], positions: [0, 100000, -100000, 0, -100000, 100000]}'
  ros2 topic pub -1 /multi_set_position \
   multi_dynamixel_interfaces/msg/MultiSetPosition \
    '{ids: [2, 4], positions: [-75937, 90000]}'
  python3 "$SLEEP"
done
