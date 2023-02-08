import RPi.GPIO as GPIO
import time

outpin = [10,24,23,22,18, 27]
inpin = 17
pi_pwm = []

for x in range (len(outpin)):
    pi_pwm.append(0)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for x in range (len(outpin)):
    GPIO.setup(outpin[x],GPIO.OUT)
    pi_pwm[x] = GPIO.PWM(outpin[x], 2000)

GPIO.setup(inpin,GPIO.IN)

oldinref: list[bool] = []
for x in range (20):
    oldinref.append(False)

oldin = oldinref

removed = False
oldinand = False
counter = 0
mult = 33
duty = [0,1,40,100]
dutyindex = 0

for x in range(len(outpin)):
    pi_pwm[x].start(duty[0])

while (True):
    removed = oldin.pop(0)
    oldin.append(GPIO.input(inpin))

    oldinand = True
    for x in oldin:
        oldinand = oldinand and x

    counter = counter + 1

    if counter == 5:
        if oldinand:
            if dutyindex == len(duty)-1:
                dutyindex = -1
            dutyindex = dutyindex + 1
            oldin[len(oldin)-1] = False
            time.sleep(1)

        counter = 0

        for x in range (len(outpin)):
            pi_pwm[x].ChangeDutyCycle(duty[dutyindex])

    time.sleep(0.05)
    print("current duty: {}".format(duty[dutyindex]))



    """
    if oldinand:
        print("LED on")
        GPIO.output(outpin,GPIO.HIGH)
        time.sleep(0.1)
    else:
        print("LED off")
        GPIO.output(outpin,GPIO.LOW)
        time.sleep(0.1)
    """
    