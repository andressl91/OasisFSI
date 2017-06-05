import sys
a = sys.argv[1]
b = sys.argv[2]
a = float(a)
b = float(b)
#print a, b
#print 100*a, 100*b
#print "E=2.0", "check=1.0"
for i in range(1, 3):
    print "E=%g" % a, "check=%g" % b
