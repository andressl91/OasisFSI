def myfunc2(*args, **kwargs):
   for a in args:
       print a
   for k,v in kwargs.iteritems():
       print "%s = %s" % (k, v)

myfunc2(1, 2, 3, banan=123)
