from tabulate import tabulate
import matplotlib.pyplot as plt

def results(Lift, Drag, Time, nel, ndof, v_deg, p_deg, cfd):

    if cfd != "cfd3":

        print "#################################### - Numerical Results - ####################################\n"
        table = []
        headers = ["Element", "nel", "ndof", "drag", "lift"]
        #headers = ["N"]
        li = []
        li.append("P%d - P%d" % (v_deg, p_deg))
        li.append(nel)
        li.append(ndof)
        li.append(Drag[-1])
        li.append(Lift[-1])
        table.append(li)

        string = tabulate.tabulate(table, headers)

        print string

        string = tabulate.tabulate(table, headers)

        text_file = open("./results/" + cfd + "/Results.txt", "w" )
        text_file.write(string)
        text_file.close()

    else:
        print "#################################### - Numerical Results - ####################################\n"

        top = []
        count = 0
        for i in range(1, len(Lift)-1):
            if Lift[i-1] < Lift[i] > Lift[i+1] and count < 2:
                top.append(Lift[i])
                count += 1

        print top

        Mean_lift = 0.5*(max(Lift) + min(Lift))
        Mean_drag = 0.5*(max(Drag) + min(Drag))

        Amp_lift = 0.5*(max(Lift) - min(Lift))
        Amp_drag = 0.5*(max(Drag) - min(Drag))

        table = []
        headers = ["Element", "nel", "ndof", "drag", "lift"]
        #headers = ["N"]
        li = []
        li.append("P%d - P%d" % (v_deg, p_deg))
        li.append(nel)
        li.append(ndof)
        li.append("%g +/- %g [%g]" % (Mean_drag, Amp_drag, 1))
        li.append("%g +/- %g [%g]" % (Mean_lift, Amp_lift, 1))
        table.append(li)

        string = tabulate.tabulate(table, headers)

        print string

        string = tabulate.tabulate(table, headers)

        text_file = open("./results/" + cfd + "/Results.txt", "w" )
        text_file.write(string)
        text_file.close()

        plt.figure(1)
        plt.plot(Time, Lift, label = "Drag")
        plt.legend()
        plt.savefig("./results/" + cfd + "/Lift.png")
        plt.figure(2)
        plt.plot(Time, Drag, label = "Drag")
        plt.legend()
        plt.savefig("./results/" + cfd + "/Drag.png")
        #plt.show()
