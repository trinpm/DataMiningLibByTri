import Utils as ut
import pandas as pd

def main():
    # url = "/home/student/trinpm/SPECS/bio/alcohol/network/miRNA_network/pfc_mirna_control.txt"
    url = "/home/student/trinpm/SPECS/bio/alcohol/network/miRNA_network/pfc_mirna_uncontrol.txt"

    cut_offs = [0.1, 0.3, 0.5, 0.7, 0.9]

    df = pd.read_csv(url, header=None, sep = "\t")
    print df.head(5)
    data = df.values

    c1 = data[:,0]
    c2 = data[:,1]
    c3 = data[:,2]

    for cut_off in cut_offs:
        if (len(c1) == len(c2) == len(c3)):
            l = []
            for i in range(0, len(c3)):
                if (c3[i] >= cut_off):
                    out = c1[i] + "\t" + c2[i] + "\t" + str(c3[i])
                    l.append(out)

        ut.writeList2file(url + ".cutoff." + str(cut_off), l)

if __name__ == '__main__':
    main()