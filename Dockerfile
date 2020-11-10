###
# AUTHOR 
#   Rui Frazão <ruifilipefrazao@ua.pt>
# 
# BUILD 
#   docker build -t nonverbal_communication_analysis . [--no-cache]
# RUN 
#   docker run -it \
#              -v /tmp/.X11-unix:/tmp/.X11-unix \
#              -e DISPLAY=$DISPLAY \ 
#              -v /path/to/synced_videos/:/nonverbal_communication_analysis/DATASET_DEP/SYNC/ \
#              -v /path/to/OPENPOSE/:/nonverbal_communication_analysis/DATASET_DEP/OPENPOSE/ \
#              -v /path/to/OPENFACE/:/nonverbal_communication_analysis/DATASET_DEP/OPENFACE/ \
#              -v /path/to/DENSEPOSE/:/nonverbal_communication_analysis/DATASET_DEP/DENSEPOSE/ \
#              -v /path/to/VIDEO/:/nonverbal_communication_analysis/DATASET_DEP/VIDEO/ \
#              nonverbal_communication_analysis \
#              <command>
# COMMANDS
#   bash -> Access Container
#   python3 nonverbal_communication_analysis/mX_modulename/file.py -h -> Commands template showing help prompt
#   Ex.: python3 nonverbal_communication_analysis/m3_DataPreProcessing/data_cleaning.py -op -of -dp DATASET_DEP/SYNC/3CLC9VWR -> Data cleaning on group 3CLC9VWR
#   python3 nonverbal_communication_analysis/m6_Visualization/visualizer.py -> Visualization tool
#
# NOTES
#   Module 1 -> Video Synchronization. Not needed for the given dataset anymore.
#   Module 2 -> Feature Extraction. Was ran on dedicated machine at IEETA.
# 
#   The remaining modules must have the result of their previous modules.
#   This container must be started with volumes linked to the experiment videos (SYNC), 
#   OpenPose (OPENPOSE), OpenFace (OPENFACE) and DensePose (DENSEPOSE) keypoints extraction files
#
###

FROM debian:buster-slim

RUN adduser --quiet --disabled-password qtuser

RUN apt update -y && apt upgrade -y
RUN apt install -y git vim python3 python3-dev python3-pip python3-pyqt5 libgl1-mesa-glx libopencv-dev python-opencv
RUN pip3 install --upgrade pip 

RUN git clone https://github.com/fraza0/nonverbal_communication_analysis.git

ENV PYTHONPATH=:~/nonverbal_communication_analysis/
WORKDIR /nonverbal_communication_analysis/

RUN pip3 install -r requirements.txt

ENV QT_QPA_PLATFORM=vnc
# CMD [ "python3 nonverbal_communication_analysis/m6_Visualization/visualizer.py" ]