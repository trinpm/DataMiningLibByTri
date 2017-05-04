import pandas

def loadData(url, d):
    df = pandas.read_csv(url, delimiter=d)
    return df

def checkGene(colA, colB):
    cnt = 0
    for geneA in colA:
        # print "---"
        # print geneA
        # print "---"
        idx_B = 0
        for geneB in colB:
            if geneA == geneB:
                #print "gene matched!, gene = ", geneA
                # print (geneA + " " + str(idx_B))
                print (str(geneA) + " " + str(idx_B))
                cnt = cnt + 1
                break
            idx_B = idx_B + 1

    print ("total number of matched genes: " + str(cnt))


def main():
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFC_gene_results.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/NAC_gene_results.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFC_NAC_Vla_Per_cons.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFC_NAC_Vla_Spearman_cons.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFCmiRNA_NACmiRNA_Joseph_Pearson_cons.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFCmiRNA_NACmiRNA_Joseph_Spearman_cons.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/PFC_miRNA_results.csv", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/NAC_miRNA_results.csv", ",")
   df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/pfc_mirna.1", ",")
   # df = loadData("/home/student/trinpm/SPECS/bio/alcohol/data/test.csv", ",")
   print df.head(5)
   print df.tail(5)

   arr = df.values
   col1 = arr[:, 0]
   col2 = arr[:, 1]
   col3 = arr[:, 2]  # list from Dr.Vladimir

   print col1[0:5]
   print col2[0:5]
   print col3[0:5]

   print "col1.shape: ", col1.shape
   print "col2.shape: ", col2.shape
   print "col3.shape: ", col3.shape

   print "-----------------------"
   print "col1 vs. col3:"
   checkGene(col3, col1)

   print "col2 vs. col3:"
   checkGene(col3, col2)
   print "-----------------------"


if __name__ == '__main__':
    main()