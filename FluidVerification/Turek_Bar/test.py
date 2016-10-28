import os

count = 1
while os.path.exists("./experiments/cfd3/"+str(count)):
    count+= 1

os.makedirs("./experiments/cfd3/"+str(count))
print count
def write():
    print("Creating report file ./experiments/cfd3/"+str(count)+"/report.txt")
    Re = 1; N = 1; dt = 1; T  = 2; v_deg =3; p_deg = 2; solver = "test"; save_step = 2; fintime = 2
    name = "./experiments/cfd3/"+str(count)+"/report.txt"  # Name of text file coerced with +.txt
    f = open(name, 'w')
    f.write("""CFD3 Turek parameters
Re = %(Re)g \nN = %(N)g\nT = %(T)g\ndt = %(dt)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\nsolver = %(solver)s\nsave_step %(save_step)s\n""" % vars())
    f.write("""Runtime = %f """ % fintime)
    f.close()


write()
