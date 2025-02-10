from test import texttonum
ob=texttonum("hello is this training class,going good?")
ob.cleaner()
ob.token()
ob.removestop()
dt=ob.stem()
print(dt)