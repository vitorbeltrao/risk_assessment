FROM jenkins/jenkins
USER root
RUN apt-get update
RUN apt-get install -y python3-pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# In this file we are doing:
# 1. Create a jenkins based image
# 2. Let's access as super user: root
# 3. Let's update the repositories
# 4. Let's install python and pip
# 5. Let's copy the .txt file to our image
# 6. Finally, we will install the .txt

# To see what images do you have, run in terminal: "docker image ls"
# To remove the an image, run in terminal: "docker image rm <IMAGE ID>"