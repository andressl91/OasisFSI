import sys, os

def userinput(args):
    arguemnt = args
    for name in vars().keys():
        print(name)
    print "print"
    if args.list:
        os.listdir("./Problems")
        for file in os.listdir("./Problems"):
            if file.endswith(".py") and file != ("__init__.py"):
                print file

    #for i in args:
    #    print i
