FROM fairnle/nle-focal:stable

RUN pip install minihack

WORKDIR /app
COPY . .
CMD [ "python3", "-m minihack.scripts.play", "--env MiniHack-River-v0" ]