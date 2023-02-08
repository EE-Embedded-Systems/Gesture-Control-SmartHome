import RPi.GPIO as GPIO
import time

outpin = 18
inpin = 17

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(outpin,GPIO.OUT)
GPIO.setup(inpin,GPIO.IN)

oldin = [False, False, False, False, False]
removed = False
oldinand = False
mult = 33
pi_pwm = GPIO.PWM(outpin, 8000)
duty = 0

pi_pwm.start(duty)

while (True):
    removed = oldin.pop(0)
    oldin.append(GPIO.input(inpin))

    oldinand = True
    for x in oldin:
        oldinand = oldinand and x
    
    if oldinand:
        duty = duty + mult
        if duty == 99:
            time.sleep(0.5)
            mult = -mult
        if duty == 0:
            time.sleep(0.5)
            mult = -mult

    pi_pwm.ChangeDutyCycle(duty)

    time.sleep(0.2)
    print("current duty: {}".format(duty))



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
    