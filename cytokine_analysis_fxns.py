import os
import glob
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Hardcoded panel-to-biomarker mapping
PANEL_MAP = {
    "Oncology": [
        "AARSD1",
        "ABL1",
        "ACAA1",
        "ACP6",
        "ADAMTS15",
        "ADAMTS8",
        "ADCYAP1R1",
        "ADGRG1",
        "ADM",
        "AGR3",
        "AIF1",
        "AIFM1",
        "AKR1B1",
        "AKT3",
        "ALPP",
        "AMBP",
        "AMIGO2",
        "ANGPT2",
        "ANGPTL7",
        "ANKRD54",
        "APBB1IP",
        "APEX1",
        "AREG",
        "ARG1",
        "ARHGAP1",
        "ARHGAP25",
        "ARSB",
        "ATG4A",
        "ATOX1",
        "ATP6AP2",
        "ATP6V1D",
        "BAIAP2",
        "BAMBI",
        "BGN",
        "BIRC2",
        "BTC",
        "C4BPB",
        "CA11",
        "CA12",
        "CA14",
        "CA9",
        "CALB1",
        "CALCOCO1",
        "CAMKK1",
        "CAPG",
        "CASP8",
        "CBLN4",
        "CCL8",
        "CCN1",
        "CCN4",
        "CCT5",
        "CD1C",
        "CD207",
        "CD27",
        "CD28",
        "CD300E",
        "CD300LF",
        "CD302",
        "CD33",
        "CD38",
        "CD5",
        "CDC27",
        "CDC37",
        "CDHR2",
        "CDKN1A",
        "CDKN2D",
        "CDNF",
        "CEACAM1",
        "CEACAM3",
        "CEACAM5",
        "CEP20",
        "CEP85",
        "CES2",
        "CES3",
        "CFC1",
        "CHAC2",
        "CIAPIN1",
        "CLEC6A",
        "CLMP",
        "CNPY4",
        "CNTN2",
        "COX5B",
        "CPE",
        "CPVL",
        "CPXM1",
        "CRACR2A",
        "CREG1",
        "CRH",
        "CRISP2",
        "CRNN",
        "CTSF",
        "CTSV",
        "CXCL8",
        "DAB2",
        "DCBLD2",
        "DCTN1",
        "DCTN2",
        "DCXR",
        "DDAH1",
        "DDX58",
        "DEFB4A_DEFB4B",
        "DKKL1",
        "DLL1",
        "DNAJB1",
        "DPEP2",
        "DPP6",
        "DPY30",
        "DRG2",
        "DSG3",
        "DSG4",
        "DTX3",
        "EBI3_IL27",
        "EDA2R",
        "EGFL7",
        "ELOA",
        "ENTPD2",
        "EPHA2",
        "EPS8L2",
        "ERBB2",
        "ERBB4",
        "ERBIN",
        "ERP44",
        "F3",
        "FAM3B",
        "FCGR2B",
        "FCRLB",
        "FEN1",
        "FES",
        "FGF21",
        "FGF23",
        "FGFBP1",
        "FGFR2",
        "FLI1",
        "FLT1",
        "FLT3",
        "FLT4",
        "FMR1",
        "FOLR1",
        "FOLR3",
        "FOXO3",
        "FURIN",
        "FUS",
        "FXN",
        "GALNT10",
        "GALNT2",
        "GALNT7",
        "GCG",
        "GCNT1",
        "GFAP",
        "GFER",
        "GFOD2",
        "GFRA1",
        "GFRA2",
        "GH2",
        "GNE",
        "GPA33",
        "GPC1",
        "GRPEL1",
        "GSAP",
        "GSTA3",
        "HAGH",
        "HAO1",
        "HAVCR1",
        "HBEGF",
        "HBQ1",
        "HDGF",
        "HGS",
        "HMBS",
        "HPGDS",
        "HS3ST3B1",
        "HS6ST1",
        "HSPB6",
        "HTRA2",
        "ICOSLG",
        "IDUA",
        "IGF1R",
        "IGSF3",
        "IKZF2",
        "IL12A_IL12B",
        "IL13RA1",
        "IL6",
        "INPP1",
        "INPPL1",
        "IQGAP2",
        "ITGAV",
        "ITGB1BP1",
        "ITGB5",
        "ITGB7",
        "KAZALD1",
        "KDR",
        "KIFBP",
        "KIR2DL3",
        "KIR3DL1",
        "KLK1",
        "KLK10",
        "KLK11",
        "KLK12",
        "KLK13",
        "KLK14",
        "KLK4",
        "KLK6",
        "KLK8",
        "KRT18",
        "L1CAM",
        "LAG3",
        "LAT2",
        "LEFTY2",
        "LGALS7_LGALS7B",
        "LHB",
        "LPCAT2",
        "LRIG1",
        "LRP1",
        "LRRC25",
        "LSM1",
        "LTA4H",
        "LTBP3",
        "LYAR",
        "LYN",
        "LYPD3",
        "LYPD8",
        "MAEA",
        "MAGED1",
        "MANSC1",
        "MAP3K5",
        "MAVS",
        "MDK",
        "MED18",
        "METAP2",
        "MIA",
        "MME",
        "MMP12",
        "MOG",
        "MPI",
        "MSLN",
        "MSRA",
        "MUC16",
        "MZT1",
        "NAMPT",
        "NBL1",
        "NCS1",
        "NDUFS6",
        "NECTIN4",
        "NELL1",
        "NFKBIE",
        "NINJ1",
        "NPTN",
        "NPY",
        "NT5E",
        "NTF4",
        "NUCB2",
        "NUDT2",
        "OGFR",
        "OMG",
        "OPTC",
        "P4HB",
        "PDCD1",
        "PDCD1LG2",
        "PDGFC",
        "PDP1",
        "PFKFB2",
        "PLA2G15",
        "PLXDC1",
        "PODXL",
        "PODXL2",
        "POLR2F",
        "PPM1A",
        "PPME1",
        "PPP1R12A",
        "PPY",
        "PQBP1",
        "PRDX6",
        "PRKRA",
        "PRTG",
        "PSMA1",
        "PSMD9",
        "PSRC1",
        "PVALB",
        "RABEPK",
        "RAD23B",
        "RANGAP1",
        "RARRES1",
        "RASSF2",
        "RBP2",
        "RBP5",
        "RET",
        "RILP",
        "RNF41",
        "RP2",
        "RRM2",
        "RRM2B",
        "RSPO3",
        "RTBDN",
        "RTN4R",
        "RUVBL1",
        "S100A12",
        "S100A4",
        "SCAMP3",
        "SCG2",
        "SCLY",
        "SCP2",
        "SEMA4C",
        "SEPTIN9",
        "SERPINA9",
        "SEZ6L",
        "SEZ6L2",
        "SF3B4",
        "SFTPA1",
        "SFTPA2",
        "SH2B3",
        "SIAE",
        "SIGLEC6",
        "SIGLEC9",
        "SIRT2",
        "SLAMF6",
        "SLAMF8",
        "SLITRK2",
        "SMAD1",
        "SMAD5",
        "SMOC1",
        "SNAP29",
        "SORCS2",
        "SORD",
        "SPARC",
        "SPINK6",
        "SRC",
        "SRP14",
        "ST3GAL1",
        "STAT5B",
        "STX16",
        "STX4",
        "STX6",
        "STXBP3",
        "SUGT1",
        "TACC3",
        "TACSTD2",
        "TAFA5",
        "TBC1D23",
        "TBL1X",
        "TEK",
        "TFPI2",
        "TGFBR2",
        "TJAP1",
        "TMPRSS15",
        "TNF",
        "TNFRSF12A",
        "TNFRSF19",
        "TP53",
        "TPMT",
        "TRIAP1",
        "TXNDC15",
        "UBAC1",
        "USO1",
        "UXS1",
        "VAT1",
        "VEGFC",
        "VMO1",
        "VNN2",
        "VPS37A",
        "VPS53",
        "VTCN1",
        "VWA1",
        "WFDC12",
        "WFDC2",
        "WIF1",
        "XCL1",
        "XPNPEP2",
        "YES1",
        "ZBTB16",
    ],
    "Neurology": [
        "ABHD14B",
        "ACVRL1",
        "ADAM22",
        "ADAM8",
        "ADGRB3",
        "AFP",
        "AGR2",
        "AHSP",
        "AKT1S1",
        "ALDH1A1",
        "AMFR",
        "ANXA10",
        "ANXA3",
        "ANXA5",
        "APOH",
        "APP",
        "APRT",
        "ARID4B",
        "ARSA",
        "ASAH2",
        "ASGR1",
        "ATF2",
        "ATP5PO",
        "ATP6V1F",
        "ATXN10",
        "B4GAT1",
        "BAG3",
        "BAX",
        "BCAM",
        "BCAN",
        "BIN2",
        "BLVRB",
        "BMP4",
        "BRK1",
        "BST1",
        "BST2",
        "C19orf12",
        "C2CD2L",
        "CA2",
        "CA6",
        "CALB2",
        "CALCA",
        "CARHSP1",
        "CASP1",
        "CASP10",
        "CC2D1A",
        "CCL19",
        "CCL2",
        "CCN5",
        "CCS",
        "CD109",
        "CD164",
        "CD177",
        "CD274",
        "CD300C",
        "CD300LG",
        "CD34",
        "CD63",
        "CD74",
        "CD8A",
        "CD99",
        "CD99L2",
        "CDCP1",
        "CDH15",
        "CDH3",
        "CDHR1",
        "CERT",
        "CETN2",
        "CGA",
        "CHGB",
        "CHMP1A",
        "CLEC10A",
        "CLEC11A",
        "CLEC14A",
        "CLEC1B",
        "CLPP",
        "CLPS",
        "CLSPN",
        "CLSTN1",
        "CNTN3",
        "CNTN4",
        "CNTN5",
        "COPE",
        "CPA2",
        "CPM",
        "CPPED1",
        "CRADD",
        "CRIP2",
        "CRTAM",
        "CSF2RA",
        "CST5",
        "CTRB1",
        "CTSS",
        "CX3CL1",
        "CXCL11",
        "CXCL13",
        "CXCL8",
        "DARS1",
        "DBI",
        "DCTN6",
        "DDR1",
        "DKK1",
        "DKK4",
        "DNMBP",
        "DPEP1",
        "DRAXIN",
        "DSC2",
        "DSG2",
        "DUSP3",
        "EBAG9",
        "ECE1",
        "EFNA1",
        "EFNA4",
        "EIF4B",
        "ENO1",
        "ENO2",
        "EPHA10",
        "EPHB6",
        "EREG",
        "EZR",
        "F11R",
        "FABP5",
        "FCER2",
        "FCRL5",
        "FGR",
        "FHIT",
        "FKBP4",
        "FKBP5",
        "FKBP7",
        "FLRT2",
        "FMNL1",
        "FOLR2",
        "FOSB",
        "FRZB",
        "FUT3_FUT5",
        "FUT8",
        "FYB1",
        "GBP4",
        "GDNF",
        "GFRA3",
        "GGA1",
        "GGT1",
        "GGT5",
        "GHRHR",
        "GKN1",
        "GLB1",
        "GLT8D2",
        "GNLY",
        "GOLM2",
        "GP6",
        "GPC5",
        "GPKOW",
        "GRN",
        "GSTP1",
        "GUCA2A",
        "HARS1",
        "HAVCR2",
        "HMOX2",
        "HNMT",
        "HSP90B1",
        "IDI2",
        "IFNGR2",
        "IFNL1",
        "IGF2R",
        "IGFBP4",
        "IL17RA",
        "IL18RAP",
        "IL1R1",
        "IL1RAP",
        "IL34",
        "IL6",
        "IL7R",
        "ILKAP",
        "IMPA1",
        "ING1",
        "INHBC",
        "IPCEF1",
        "ISLR2",
        "ITGA5",
        "ITGAM",
        "IVD",
        "JAM2",
        "KCNIP4",
        "KEL",
        "KIRREL2",
        "KLB",
        "KRT14",
        "KRT5",
        "LAIR2",
        "LAMP2",
        "LAYN",
        "LBR",
        "LGALS8",
        "LIF",
        "LILRA2",
        "LPO",
        "LRPAP1",
        "LXN",
        "LY96",
        "LYPD1",
        "MAD1L1",
        "MAP4K5",
        "MAPT",
        "MASP1",
        "MATN3",
        "MAX",
        "MDGA1",
        "MESD",
        "METAP1",
        "MFGE8",
        "MIF",
        "MITD1",
        "MMP13",
        "MMP3",
        "MMP8",
        "MMP9",
        "MPO",
        "MRPL46",
        "MSR1",
        "MUC13",
        "MYOC",
        "NAAA",
        "NCAM2",
        "NCAN",
        "NDRG1",
        "NEFL",
        "NGF",
        "NID2",
        "NMNAT1",
        "NOMO1",
        "NOS1",
        "NOS3",
        "NPM1",
        "NPTX1",
        "NRP2",
        "NSFL1C",
        "NTRK3",
        "NUDT5",
        "NXPH1",
        "OBP2B",
        "ODAM",
        "OGN",
        "OXT",
        "PADI4",
        "PAEP",
        "PAK4",
        "PAMR1",
        "PARK7",
        "PBLD",
        "PDCD5",
        "PEBP1",
        "PECAM1",
        "PFDN2",
        "PHOSPHO1",
        "PIGR",
        "PIK3IP1",
        "PILRA",
        "PLA2G10",
        "PLA2G7",
        "PLAU",
        "PLIN1",
        "PMVK",
        "PPCDC",
        "PPP3R1",
        "PRDX1",
        "PRL",
        "PRTFDC1",
        "PSG1",
        "PSME1",
        "PSME2",
        "PTEN",
        "PTK7",
        "PTPN1",
        "PTPRN2",
        "PTS",
        "PVR",
        "PXN",
        "RAB6B",
        "RASA1",
        "RBKS",
        "RELT",
        "RGMA",
        "RGMB",
        "RHOC",
        "ROBO2",
        "RSPO1",
        "RWDD1",
        "S100A16",
        "SCARA5",
        "SCARB1",
        "SCARB2",
        "SCARF2",
        "SEMA4D",
        "SERPINB1",
        "SERPINB6",
        "SERPINB9",
        "SESTD1",
        "SETMAR",
        "SFRP1",
        "SIGLEC15",
        "SIGLEC5",
        "SIRT5",
        "SKAP1",
        "SLC16A1",
        "SLC27A4",
        "SLC39A14",
        "SLIT2",
        "SMARCA2",
        "SMPD1",
        "SNCG",
        "SOD2",
        "SPINK1",
        "SPINK5",
        "SPINT1",
        "SPOCK1",
        "SSB",
        "STAMBP",
        "STC1",
        "STC2",
        "STIP1",
        "STK24",
        "SULT1A1",
        "SUMF2",
        "SUSD2",
        "TARBP2",
        "TBC1D17",
        "TBCB",
        "TBCC",
        "TCL1A",
        "TDGF1",
        "TDRKH",
        "TFF1",
        "THBS2",
        "THY1",
        "TIGAR",
        "TIMP4",
        "TMPRSS5",
        "TMSB10",
        "TNF",
        "TNFRSF10A",
        "TNFRSF10B",
        "TNFRSF1A",
        "TNFRSF1B",
        "TNFRSF21",
        "TNFRSF6B",
        "TNFRSF8",
        "TNFRSF9",
        "TNFSF14",
        "TNR",
        "TNXB",
        "TPPP3",
        "TREML2",
        "TST",
        "TXLNA",
        "TXNDC5",
        "TXNRD1",
        "ULBP2",
        "VCAN",
        "VSIG4",
        "VSTM1",
        "VTA1",
        "VWC2",
        "WARS",
        "WASF3",
        "WFIKKN1",
        "WWP2",
        "XRCC4",
    ],
    "Cardiometabolic": [
        "ACAN",
        "ACE2",
        "ACOX1",
        "ACP5",
        "ACTA2",
        "ACY1",
        "ADA2",
        "ADAM15",
        "ADAMTS13",
        "ADAMTS16",
        "ADGRE5",
        "ADGRG2",
        "ADH4",
        "AGXT",
        "AHCY",
        "AK1",
        "AKR1C4",
        "ALCAM",
        "AMY2A",
        "AMY2B",
        "ANG",
        "ANGPTL1",
        "ANGPTL3",
        "ANPEP",
        "ANXA4",
        "AOC3",
        "APLP1",
        "APOM",
        "ART3",
        "AXL",
        "AZU1",
        "BAG6",
        "BLMH",
        "BMP6",
        "BOC",
        "BPIFB1",
        "C1QTNF1",
        "C2",
        "CA1",
        "CA13",
        "CA3",
        "CA4",
        "CA5A",
        "CANT1",
        "CASP3",
        "CBLIF",
        "CCDC80",
        "CCL14",
        "CCL15",
        "CCL16",
        "CCL18",
        "CCL27",
        "CCL5",
        "CCN3",
        "CD14",
        "CD163",
        "CD209",
        "CD2AP",
        "CD46",
        "CD55",
        "CD59",
        "CD69",
        "CD93",
        "CDH1",
        "CDH17",
        "CDH2",
        "CDH5",
        "CDH6",
        "CDHR5",
        "CEACAM8",
        "CEBPB",
        "CELA3A",
        "CEP43",
        "CES1",
        "CGREF1",
        "CHEK2",
        "CHI3L1",
        "CHIT1",
        "CHL1",
        "CHRDL2",
        "CLC",
        "CLEC1A",
        "CLEC5A",
        "CLTA",
        "CLUL1",
        "CNDP1",
        "CNPY2",
        "CNST",
        "CNTN1",
        "COL18A1",
        "COL1A1",
        "COL4A1",
        "COL6A3",
        "COMP",
        "COMT",
        "CORO1A",
        "CPA1",
        "CPB1",
        "CR2",
        "CRHR1",
        "CRTAC1",
        "CRX",
        "CST3",
        "CST6",
        "CSTB",
        "CTF1",
        "CTSB",
        "CTSD",
        "CTSH",
        "CTSL",
        "CTSZ",
        "CXCL16",
        "CXCL5",
        "CXCL8",
        "DCN",
        "DCTPP1",
        "DDC",
        "DEFA1_DEFA1B",
        "DIABLO",
        "DKK3",
        "DLK1",
        "DNAJB8",
        "DOK2",
        "DPP4",
        "DPP7",
        "DPT",
        "DUOX2",
        "EDIL3",
        "EFEMP1",
        "EGFR",
        "EIF4EBP1",
        "ENG",
        "ENPP2",
        "ENTPD5",
        "ENTPD6",
        "EPHB4",
        "EPHX2",
        "ESAM",
        "F7",
        "F9",
        "FABP2",
        "FABP4",
        "FABP6",
        "FADD",
        "FAM3C",
        "FAP",
        "FAS",
        "FBP1",
        "FCGR2A",
        "FCGR3B",
        "FCN2",
        "FCRL1",
        "FETUB",
        "FUCA1",
        "GAS6",
        "GDF15",
        "GDF2",
        "GGH",
        "GH1",
        "GHRL",
        "GLO1",
        "GLRX",
        "GP1BA",
        "GP2",
        "GPNMB",
        "GPR37",
        "GRAP2",
        "GRK5",
        "GSTA1",
        "GUSB",
        "GYS1",
        "GZMH",
        "HEBP1",
        "HK2",
        "HMOX1",
        "HNRNPK",
        "HSPB1",
        "HSPG2",
        "HYAL1",
        "HYOU1",
        "ICAM1",
        "ICAM2",
        "ICAM3",
        "ICAM5",
        "IGFBP1",
        "IGFBP2",
        "IGFBP3",
        "IGFBP6",
        "IGFBP7",
        "IGFBPL1",
        "IGSF8",
        "IL18BP",
        "IL19",
        "IL1RL1",
        "IL2RA",
        "IL6",
        "IL6R",
        "IL6ST",
        "IRAG2",
        "ITGB1",
        "ITGB1BP2",
        "ITGB2",
        "ITIH3",
        "KIT",
        "KITLG",
        "KYAT1",
        "LACTB2",
        "LBP",
        "LCN2",
        "LDLR",
        "LEP",
        "LEPR",
        "LGALS1",
        "LGALS3",
        "LILRA5",
        "LILRB1",
        "LILRB2",
        "LILRB5",
        "LPL",
        "LRP11",
        "LTBP2",
        "MARCO",
        "MB",
        "MCAM",
        "MCFD2",
        "MEGF9",
        "MEP1B",
        "MET",
        "MFAP3",
        "MFAP5",
        "MMP7",
        "MNDA",
        "MPHOSPH8",
        "MSMB",
        "MSTN",
        "MTPN",
        "NADK",
        "NCAM1",
        "NECTIN2",
        "NID1",
        "NOTCH1",
        "NOTCH3",
        "NPDC1",
        "NPPB",
        "NPTXR",
        "NRCAM",
        "NRP1",
        "NTRK2",
        "NTproBNP",
        "OLR1",
        "OSMR",
        "PAG1",
        "PAM",
        "PCDH17",
        "PCOLCE",
        "PCSK9",
        "PDCD6",
        "PDGFA",
        "PDGFRA",
        "PDGFRB",
        "PEAR1",
        "PGLYRP1",
        "PI3",
        "PILRB",
        "PLA2G1B",
        "PLA2G2A",
        "PLAT",
        "PLIN3",
        "PLPBP",
        "PLTP",
        "PLXNB2",
        "PLXNB3",
        "PM20D1",
        "PON2",
        "PPIB",
        "PPP1R2",
        "PRCP",
        "PRKAR1A",
        "PROC",
        "PRSS2",
        "PRSS27",
        "PRTN3",
        "PTGDS",
        "PTN",
        "PTPRF",
        "PTPRS",
        "QDPR",
        "QPCT",
        "RARRES2",
        "RCOR1",
        "REG1A",
        "REG1B",
        "REG3A",
        "REN",
        "RETN",
        "RNASE3",
        "RNASET2",
        "ROR1",
        "S100A11",
        "S100P",
        "SCARF1",
        "SDC1",
        "SDC4",
        "SELE",
        "SELP",
        "SEMA3F",
        "SEMA7A",
        "SERPINA11",
        "SERPINA12",
        "SERPINB5",
        "SERPINE1",
        "SFTPD",
        "SIGLEC7",
        "SIRPA",
        "SLITRK6",
        "SNAP23",
        "SNX9",
        "SOD1",
        "SORT1",
        "SOST",
        "SPARCL1",
        "SPON2",
        "SPP1",
        "SSC4D",
        "SSC5D",
        "ST6GAL1",
        "STK11",
        "STK4",
        "SUSD1",
        "TCL1B",
        "TCN2",
        "TFF3",
        "TFPI",
        "TFRC",
        "TGFBI",
        "TGFBR3",
        "TGM2",
        "THBD",
        "THBS4",
        "THOP1",
        "THPO",
        "TIA1",
        "TIE1",
        "TIMD4",
        "TIMP1",
        "TINAGL1",
        "TNC",
        "TNF",
        "TNFRSF10C",
        "TNFSF13B",
        "TNNI3",
        "TP53INP1",
        "TSHB",
        "TSLP",
        "TSPAN1",
        "TYMP",
        "TYRO3",
        "UMOD",
        "USP8",
        "VAMP5",
        "VASN",
        "VCAM1",
        "VIM",
        "VSIR",
        "VSTM2L",
        "VWF",
        "WASF1",
        "XG",
        "ZBTB17",
    ],
    "Inflammation": [
        "ACTN4",
        "ADA",
        "ADAM23",
        "ADGRE2",
        "AGER",
        "AGRN",
        "AGRP",
        "ALDH3A1",
        "AMBN",
        "AMN",
        "ANGPT1",
        "ANGPTL2",
        "ANGPTL4",
        "ANXA11",
        "AOC1",
        "ARHGEF12",
        "ARNT",
        "ARTN",
        "ATP5IF1",
        "AXIN1",
        "B4GALT1",
        "BACH1",
        "BANK1",
        "BCL2L11",
        "BCR",
        "BID",
        "BSG",
        "BTN2A1",
        "BTN3A2",
        "C1QA",
        "CASP2",
        "CCL11",
        "CCL13",
        "CCL17",
        "CCL20",
        "CCL21",
        "CCL22",
        "CCL23",
        "CCL24",
        "CCL25",
        "CCL26",
        "CCL28",
        "CCL3",
        "CCL4",
        "CCL7",
        "CCN2",
        "CD160",
        "CD200",
        "CD200R1",
        "CD22",
        "CD244",
        "CD276",
        "CD4",
        "CD40",
        "CD40LG",
        "CD48",
        "CD58",
        "CD6",
        "CD70",
        "CD79B",
        "CD83",
        "CD84",
        "CDON",
        "CDSN",
        "CEACAM21",
        "CEP164",
        "CHRDL1",
        "CKAP4",
        "CKMT1A_CKMT1B",
        "CLEC4A",
        "CLEC4C",
        "CLEC4D",
        "CLEC4G",
        "CLEC7A",
        "CLIP2",
        "CLSTN2",
        "CNTNAP2",
        "COL9A1",
        "COLEC12",
        "CRELD2",
        "CRHBP",
        "CRIM1",
        "CRKL",
        "CRLF1",
        "CSF1",
        "CSF3",
        "CST7",
        "CTRC",
        "CTSC",
        "CTSO",
        "CXADR",
        "CXCL1",
        "CXCL10",
        "CXCL12",
        "CXCL14",
        "CXCL17",
        "CXCL3",
        "CXCL6",
        "CXCL8",
        "CXCL9",
        "DAG1",
        "DAPP1",
        "DBNL",
        "DECR1",
        "DFFA",
        "DGKZ",
        "DNAJA2",
        "DNER",
        "DNPH1",
        "DPP10",
        "EDAR",
        "EGF",
        "EGLN1",
        "EIF4G1",
        "EIF5A",
        "ENAH",
        "ENPP5",
        "ENPP7",
        "EPCAM",
        "EPHA1",
        "EPO",
        "ERBB3",
        "ESM1",
        "F2R",
        "FABP1",
        "FABP9",
        "FASLG",
        "FCAR",
        "FCRL2",
        "FCRL3",
        "FCRL6",
        "FGF19",
        "FGF2",
        "FGF5",
        "FIS1",
        "FKBP1B",
        "FLT3LG",
        "FOXO1",
        "FST",
        "FSTL3",
        "FXYD5",
        "GAL",
        "GALNT3",
        "GBP2",
        "GLOD4",
        "GMPR",
        "GOPC",
        "GZMA",
        "GZMB",
        "HCLS1",
        "HEXIM1",
        "HGF",
        "HLA-DRA",
        "HLA-E",
        "HPCAL1",
        "HSD11B1",
        "HSPA1A",
        "ICA1",
        "ICAM4",
        "IDS",
        "IFNG",
        "IFNGR1",
        "IFNLR1",
        "IKBKG",
        "IL10",
        "IL10RA",
        "IL10RB",
        "IL11",
        "IL12B",
        "IL12RB1",
        "IL13",
        "IL15",
        "IL15RA",
        "IL16",
        "IL17A",
        "IL17C",
        "IL17D",
        "IL17F",
        "IL17RB",
        "IL18",
        "IL18R1",
        "IL1A",
        "IL1B",
        "IL1R2",
        "IL1RL2",
        "IL1RN",
        "IL2",
        "IL20",
        "IL20RA",
        "IL22RA1",
        "IL24",
        "IL2RB",
        "IL32",
        "IL33",
        "IL3RA",
        "IL4",
        "IL4R",
        "IL5",
        "IL5RA",
        "IL6",
        "IL7",
        "IRAK1",
        "IRAK4",
        "ISM1",
        "ITGA11",
        "ITGA6",
        "ITGB6",
        "ITM2A",
        "JCHAIN",
        "JUN",
        "KLRB1",
        "KLRD1",
        "KRT19",
        "KYNU",
        "LAIR1",
        "LAMA4",
        "LAMP3",
        "LAP3",
        "LAT",
        "LGALS4",
        "LGALS9",
        "LGMN",
        "LHPP",
        "LIFR",
        "LILRB4",
        "LRRN1",
        "LSP1",
        "LTA",
        "LTBR",
        "LTO1",
        "LY6D",
        "LY75",
        "LY9",
        "MANF",
        "MAP2K6",
        "MAPK9",
        "MATN2",
        "MEGF10",
        "MEPE",
        "MERTK",
        "METAP1D",
        "MGLL",
        "MGMT",
        "MICB_MICA",
        "MILR1",
        "MLN",
        "MMP1",
        "MMP10",
        "MPIG6B",
        "MVK",
        "MYO9B",
        "MZB1",
        "NBN",
        "NCF2",
        "NCK2",
        "NCLN",
        "NCR1",
        "NELL2",
        "NFASC",
        "NFATC1",
        "NFATC3",
        "NME3",
        "NPPC",
        "NRTN",
        "NT5C3A",
        "NTF3",
        "NUB1",
        "NUDC",
        "OMD",
        "OSCAR",
        "OSM",
        "PADI2",
        "PAPPA",
        "PARP1",
        "PCDH1",
        "PDGFB",
        "PDLIM7",
        "PGF",
        "PIK3AP1",
        "PKLR",
        "PLA2G4A",
        "PLAUR",
        "PLXNA4",
        "PNLIPRP2",
        "PNPT1",
        "PON3",
        "PPP1R9B",
        "PRDX3",
        "PRDX5",
        "PREB",
        "PRELP",
        "PRKAB1",
        "PRKCQ",
        "PROK1",
        "PRSS8",
        "PSIP1",
        "PSMG3",
        "PSPN",
        "PTH1R",
        "PTPN6",
        "PTPRM",
        "PTX3",
        "RAB37",
        "RAB6A",
        "RABGAP1L",
        "REG4",
        "RGS8",
        "ROBO1",
        "SAMD9L",
        "SCG3",
        "SCGB1A1",
        "SCGB3A2",
        "SCGN",
        "SCRN1",
        "SELPLG",
        "SERPINB8",
        "SH2D1A",
        "SHMT1",
        "SIGLEC1",
        "SIGLEC10",
        "SIRPB1",
        "SIT1",
        "SKAP2",
        "SLAMF1",
        "SLAMF7",
        "SLC39A5",
        "SMOC2",
        "SMPDL3A",
        "SPINK4",
        "SPINT2",
        "SPON1",
        "SPRY2",
        "SRPK2",
        "STX8",
        "SULT2A1",
        "TANK",
        "TBC1D5",
        "TFF2",
        "TGFA",
        "TGFB1",
        "TIMP3",
        "TLR3",
        "TNF",
        "TNFAIP8",
        "TNFRSF11A",
        "TNFRSF11B",
        "TNFRSF13B",
        "TNFRSF13C",
        "TNFRSF14",
        "TNFRSF4",
        "TNFSF10",
        "TNFSF11",
        "TNFSF12",
        "TNFSF13",
        "TPP1",
        "TPSAB1",
        "TPT1",
        "TRAF2",
        "TREM2",
        "TRIM21",
        "TRIM5",
        "VASH1",
        "VEGFA",
        "VEGFD",
        "WAS",
        "WFIKKN2",
        "WNT9A",
        "YTHDF3",
    ],
    "Target 48": [
        "FLT3LG", "CSF3", "CSF2", "IFNG", "IL1B", "IL2", "IL4", "IL6", "IL7", "CXCL8",
        "IL10", "IL13", "IL15", "IL17A", "IL17C", "IL17F", "IL18", "IL27", "IL33", "LTA",
        "CSF1", "OSM", "TNF", "TNFSF10", "TSLP", "TNFSF12", "CCL2", "CCL3", "CCL4", "CCL7",
        "CCL8", "CCL13", "CCL19", "CXCL9", "CXCL10", "CXCL11", "CCL11", "CXCL12", "HGF",
        "MMP1", "MMP12", "OLR1", "EGF", "TGFA", "VEGFA"
    ],
    "Target 96 Cardiovascular II": [
        "DECR1", "ADAMTS13", "ADM", "AGRP", "IDUA", "ANGPT1", "TEK", "ACE2", "BMP6", "BOC", "CA5A", 
        "CEACAM8", "CTSL1", "CCL3", "CCL17", "CD40LG", "CTRC", "CXCL1", "DCN", "DKK1", "FABP2", "FGF21", 
        "FGF23", "FST", "LGALS9", "GIF", "FABP6", "GH1", "GDF2", "HSPB1", "HMOX1", "HAOX1", "IL1RN", 
        "IL1RL2", "IL4R", "IL6", "IL17D", "IL18", "IL27", "HAVCR1", "GLO1", "OLR1", "LEP", "LPL", 
        "FCGR2B", "XCL1", "MARCO", "MMP7", "MMP12", "ITGB1BP2", "NPPB", "IKBKG", "OSCAR", "PAPPA", 
        "PTX3", "PGF", "PDGFB", "PARP1", "PIGR", "PDCD1LG2", "HBEGF", "IL16", "PRELP", "PRSS8", "AMBP", 
        "F2R", "TGM2", "SRC", "SELPLG", "AGER", "REN", "PRSS27", "STK4", "SERPINA12", "CD84", "SLAMF7", 
        "SORT1", "SPON2", "KITLG", "SOD2", "CD4", "THBD", "THPO", "THBS2", "F3", "TNFRSF10B", 
        "TNFRSF10A", "TNFRSF11A", "TNFRSF13B", "MERTK", "VEGFD", "VSIG2"  
    ],
    "Target 96 Cardiovascular III": [
        "GDF15", "TNFRSF14", "LDLR", "ITGB2", "IL17RA", "TNFRSF1B", "MMP9", "EPHB4", "IL2RA", 
        "TNFRSF11B", "ALCAM", "TFF3", "SELP", "CSTB", "CCL2", "CD163", "LGALS3", "GRN", "BLMH", 
        "HSPG2", "LTBR", "NOTCH3", "TIMP4", "CNTN1", "CDH5", "TREML2", "FABP4", "TFPI", "SERPINE1", 
        "CCL24", "TFRC", "TNFRSF10C", "SELE", "AZU1", "DLK1", "SPON1", "MPO", "CXCL16", "IL6R", "RETN", 
        "IGFBP1", "IGFBP2", "IGFBP7", "CHIT1", "ACP5", "GP6", "SFTPD", "PI3", "EPCAM", "ANPEP", "AXL", 
        "IL1R1", "IL1R2", "IL1RL1", "IL18BP", "FAS", "TNFRSF1A", "TNFSF13B", "PRTN3", "PCSK9", "PLAUR", 
        "PLAU", "PLAT", "SPP1", "CTSD", "CTSZ", "CASP3", "CPA1", "CPB1", "PGLYRP1", "PON3", "COL1A1", 
        "F11R", "LGALS4", "MEPE", "MMP2", "MMP3", "RARRES2", "ICAM2", "KLK6", "PDGFA", "SCGB3A2", 
        "SIRPA", "ST2", "TRAP", "UPAR", "VWF", "MB", "MCP1", "PI3", "SFTPD", "NPPB", "AZU1"
    ],
    "Target 96 Cardiometabolic": [
        "ANG", "ANGPTL3", "AOC3", "APOM", "C1QTNF1", "C2", "CA1", "CA3", "CA4", "CCL14", "CCL18", 
        "CCL5", "CD46", "CD59", "CDH1", "CES1", "CHL1", "CNDP1", "COL18A1", "COMP", "CR2", "CRTAC1", 
        "CST3", "DPP4", "DEFA1", "EFEMP1", "ENG", "F11", "F7", "FAP", "FCGR2A", "FCGR3B", "FETUB", 
        "FCN2", "GAS6", "GNLY", "GP1BA", "ICAM1", "ICAM3", "IGFBP3", "IGFBP6", "IGLC2", "IL7R", "ITGAM", 
        "KIT", "LCN2", "LILRB1", "LILRB2", "LILRB5", "LTBP2", "LYVE1", "MBL2", "MEGF9", "MET", "MFAP5", 
        "NCAM1", "NID1", "NOTCH1", "NRP1", "OSMR", "PAM", "PCOLCE", "PLA2G7", "PLTP", "PLXNB2", "PRCP", 
        "PROC", "PRSS2", "PTPRS", "QPCT", "REG1A", "REG3A", "SAA4", "SELL", "SERPINA5", "SERPINA7", 
        "SOD1", "SPARCL1", "ST6GAL1", "TCN2", "TGFBI", "TGFBR3", "THBS4", "TIE1", "TIMD4", "TIMP1", 
        "TNC", "TNXB", "UMOD", "VASN", "VCAM1"
    ],
    "Target 96 Immuno-Oncology": [
        "ADA", "ADGRG1", "ANGPT1", "ANGPT2", "ARG1", "CA9", "CASP8", "CCL13", "CCL17", "CCL19", "CCL2", 
        "CCL20", "CCL23", "CCL3", "CCL4", "CCL7", "CCL8", "CD244", "CD27", "CD28", "CD4", "CD40", 
        "CD40LG", "CD5", "CD70", "CD8A", "CD83", "CD274", "CX3CL1", "CXCL1", "CXCL10", "CXCL11", 
        "CXCL12", "CXCL13", "CXCL5", "CXCL9", "CRTAM", "DCN", "EGF", "FGF2", "GZMA", "GZMB", "GZMH", 
        "HGF", "HMOX1", "ICOSLG", "IFNG", "IL1A", "IL10", "IL12", "IL12RB1", "IL13", "IL15", "IL18", 
        "IL2", "IL33", "IL4", "IL5", "IL6", "IL7", "IL8", "KDR", "KIR3DL1", "LAG3", "LAMP3", "LGALS1", 
        "LGALS9", "MICA", "MMP12", "MMP7", "MUC16", "NOS3", "PDCD1", "PDCD1LG2", "PDGFB", "PGF", "PTN", 
        "SNCAIP", "TGFB1", "TNF", "TNFRSF12A", "TNFRSF21", "TNFRSF4", "TNFRSF9", "TNFSF10", "TNFSF12", 
        "TNFSF14", "TRAIL", "TWEAK", "VEGFA", "XCL1"
    ],
    "Target 96 Immune Response": [
        "AREG", "ARNT", "BACH1", "BIRC2", "BTN3A2", "CCL11", "CD28", "CD83", "CKAP4", "CLEC4A", 
        "CLEC4C", "CLEC4D", "CLEC4G", "CLEC6A", "CLEC7A", "CXADR", "CXCL12", "DAPP1", "DCTN1", 
        "DCBLD2", "DFFA", "DGKZ", "DPP10", "EGLN1", "EIF4G1", "EIF5A", "FAM3B", "FCRL3", "FCRL6", 
        "FGF2", "FPRDX3", "FPRDX5", "FXYD5", "GALNT3", "GLB1", "HCLS1", "HEXIM1", "HNMT", "HSD11B1", 
        "ICA1", "IFNLR1", "IL10", "IL12RB1", "IL5", "IL6", "IL6", "IRAK1", "IRAK4", "IRF9", "ITGA11", 
        "ITGA6", "ITGB6", "ITM2A", "JUN", "KLRD1", "KPNA1", "KRT19", "LAG3", "LAMP3", "LILRB4", "LY75", 
        "MASP1", "MGMT", "MILR1", "NF2", "NFATC3", "NTF4", "PADI2", "PIK3AP1", "PLXNA4", "PRDX1", 
        "PRKCQ", "PSIP1", "PTK2", "PTH1R", "RIGI", "SH2B3", "SH2D1A", "SIT1", "SPRY2", "SRPK2", "STC1", 
        "STC1", "STC1", "TANK", "TPSAB1", "TRAF2", "TREM1", "TRIM21", "TRIM5", "ZBTB16"
    ],
    "Target 96 Inflammation": [
        "ADA", "ARTN", "AXIN1", "CASP8", "CCL11", "CCL13", "CCL19", "CCL2", "CCL20", "CCL23", "CCL25", 
        "CCL28", "CCL3", "CCL4", "CCL7", "CCL8", "CD244", "CD274", "CD40", "CD5", "CD6", "CD8A", 
        "CDCP1", "CSF1", "CST5", "CX3CL1", "CXCL1", "CXCL10", "CXCL11", "CXCL5", "CXCL6", "CXCL8", 
        "CXCL9", "DNER", "EIF4EBP1", "FGF19", "FGF21", "FGF23", "FGF5", "FLT3LG", "GDNF", "HGF", 
        "IFNG", "IL10", "IL10RA", "IL10RB", "IL12B", "IL13", "IL15RA", "IL17A", "IL17C", "IL18", 
        "IL18R1", "IL1A", "IL2", "IL20", "IL20RA", "IL22RA1", "IL24", "IL2RB", "IL33", "IL4", "IL5", 
        "IL6", "IL7", "KITLG", "LIF", "LIFR", "LTA", "MMP1", "MMP10", "NGF", "NRTN", "NTF3", "OSM", 
        "PLAU", "S100A12", "SIRT2", "SLAMF1", "STAMBP", "SULT1A1", "TGFA", "TGFB1", "TNF", "TNFRSF11B", 
        "TNFRSF9", "TNFSF10", "TNFSF11", "TNFSF12", "TNFSF14", "TSLP", "VEGFA"
    ],
    "Target 96 Neurology": [
        "ACVRL1", "ADAM22", "ADAM23", "ASAH2", "BCAN", "BMP4", "CD38", "CDH3", "CDH6",
        "CD200", "CD200R1", "CD300C", "CD300LF", "CADM3", "CLM1", "CLM6", "CLEC1B",
        "CLEC10A", "CNTN5", "CPA2", "CPM", "CRTAM", "CSF3", "CSF2RA", "CTSC", "CTSS",
        "DDR1", "DKK4", "DRAXIN", "EFNA4", "EPHB6", "EZR", "FCRL2", "FLRT2", "FRZB",
        "GAL8", "GDNF", "GFRA1", "GFRA3", "GDF8", "GPC5", "GZMA", "HAGH", "IL5RA",
        "IL12B", "JAM2", "KYNU", "LAIR2", "LAT", "LAYN", "LRPAP1", "LXN", "MANF",
        "MAPT", "MATN3", "MDGA1", "MSR1", "MME", "MSTN", "NAAA", "NCAN", "NBL1",
        "NCRAM", "NTRK2", "NTRK3", "NMNAT1", "NRP2", "OX2", "PDGFRA", "PLXNB1",
        "PLXNB3", "PRTG", "PVR", "RGMA", "RGMB", "ROBO2", "RSPO1", "SCARA5", "SCARF2",
        "SCARB2", "SIGLEC1", "SIGLEC9", "SMOC2", "SMPD1", "SPOCK1", "SFRP3", "SKR3",
        "TMPRSS5", "THY1", "TNFRSF12A", "TNFRSF21", "UNC5C", "ULBP2", "WFIKKN1"
    ],
    "Target 96 Neuro Exploratory": [
        "IFNL1", "EIF4B", "CRADD", "CRIPTO", "ECE1", "CETN2", "CDH15", "SMOC1", "ADGRB3", "KLB",
        "CDH17", "GPNMB", "BST2", "PTPN1", "SRP14", "ATP6V1F", "RBKS", "FKBP7", "ANXA10", "GSTP1",
        "KIR2DL3", "EREG", "CD302", "IMPA1", "CRIP2", "LTBP3", "CTF1", "FHIT", "CLSTN1", "HSP90B1",
        "LEPR", "CD33", "ADAM15", "NDRG1", "CEACAM3", "GBP2", "FGFR2", "NXPH1", "NAA10", "IL15",
        "DSG3", "SFRP1", "IFI30", "FCAR", "PRTFDC1", "PLA2G10", "IKZF2", "DPEP2", "TBCB", "NPM1",
        "ASGR1", "EPHA10", "CERT1", "PSG1", "PSME1", "KIRREL2", "PAEP", "CCL27", "MAD1L1", "AKT1S1",
        "PTS", "IL32", "FUT8", "TPPP3", "PFDN2", "CARHSP1", "VSTM1", "DEFB4A", "ABHD14B", "AARSD1",
        "PHOSPHO1", "DUSP3", "TNFRSF13C", "NEFL", "HMOX2", "RNF31", "DPEP1", "SNCG", "IL3RA", "AOC1",
        "KIFBP", "PPP3R1", "ILKAP", "ISLR2", "ING1", "PMVK", "WWP2", "FKBP5", "GGT5", "CD63",
        "RPS6KB1", "UBE2F"
    ],
    "Target 96 Oncology II": [
        "5NT", "ABL1", "ADAM8", "ADAMTS15", "ANXA1", "AREG", "CA9", "CEA", "CEACAM1", "CD160",
        "CD207", "CD27", "CD48", "CD70", "CDKN1A", "CRNN", "CXCL13", "CYR61", "DLL1", "EPHA2",
        "ERBB2", "ERBB3", "ERBB4", "ESM1", "FADD", "FASLG", "FCRLB", "FGFBP1", "FLT4", "FOLR1",
        "FOLR3", "FURIN", "GPC1", "GPNMB", "GZMB", "GZMH", "HGF", "ICOSLG", "IFNGR1", "IGF1R",
        "IL6", "ITGAV", "ITGB5", "KLK11", "KLK13", "KLK14", "KLK8", "LGALS1", "LYN", "LYPD3",
        "LY9", "MAD5", "MDK", "METAP2", "MIA", "MICAB", "MSLN", "MUC16", "NT5E", "PODXL",
        "PPY", "PVRL4", "RET", "RSPO3", "SCAMP3", "SCF", "SDC1", "SEZ6L", "SMAD5", "SPARC",
        "S100A11", "S100A4", "TCL1A", "TGFBR2", "TGFA", "TFPI2", "TLR3", "TNFSF10", "TNFSF13",
        "TNFRSF4", "TNFRSF6B", "TNFRSF19", "TXLNA", "VEGFA", "VIM", "WFDC2", "WIF1", "WISP1",
        "XPNPEP2", "CXCL17", "GZMB", "GPC1", "ADAMTS15"
    ],
    "Target 96 Oncology III": [
        "CPVL", "NT5C3A", "CDHR2", "UBAC1", "ERP44", "GOPC", "RFNG", "PTP4A1", "AIMP1",
        "AKR1B1", "FLT1", "CD22", "RAB6A", "COL9A1", "PTPRM", "LAP3", "CDC27", "L1CAM",
        "LSP1", "HLAE", "FUS", "IFNGR2", "ARHGAP25", "CASP2", "NAMPT", "CCT5", "INPP1",
        "TPMT", "AIF1", "DRG2", "GFER", "FLT3", "NCS1", "TPT1", "MLN", "HSPB6", "FOXO3",
        "ACTN4", "PSPN", "SPINK4", "USO1", "TBL1X", "RP2", "SCGN", "TNFAIP8", "SCG2",
        "CXCL14", "IL1B", "JCHAIN", "C1QA", "AFP", "ALPP", "HEXA", "HMBS", "HBQ1",
        "ACAA1", "CGB3_CGB5_CGB8", "NPY", "PSMD9", "SCLY", "S100A16", "NELL1", "BRK1",
        "VPS37A", "SFTPA1", "GALNT7", "YTHDF3", "VMO1", "VWA1", "VPS53", "LACTB2",
        "CD300E", "ICAM4", "GAMT", "ELOA", "MYO9B", "PCDH1", "IL17F", "TXNDC15", "GPA33",
        "ATP6V1D", "KLK4", "CNPY2", "ICAM5", "GGA1", "SEPTIN9", "CLIP2", "CALCOCO1",
        "VAT1", "CBLN4", "MANSC1", "DCTPP1"
    ],
    "Target 96 Metabolism": [
        "AHCY", "S100P", "ITGB7", "APEX1", "CD1C", "SDC4", "SERPINB6", "GLRX", "CD79B", "USP8",
        "GAL", "PPP1R2", "CRKL", "BAG6", "SERPINB8", "ANXA11", "APLP1", "HDGF", "THOP1", "DAB2",
        "ROR1", "FKBP4", "CTSO", "TFF2", "COMT", "TYMP", "SNAP23", "ANGPT2", "KLK10", "ANGPTL7",
        "ENTPD5", "GRAP2", "NPTXR", "NADK", "ANGPTL1", "DDC", "ALDH1A1", "ARG1", "ENO2", "QDPR",
        "FBP1", "ANXA4", "CTSH", "RNASE3", "VCAN", "CDH2", "TSHB", "LILRA5", "NPPB", "NQO2",
        "CANT1", "SSC4D", "MCFD2", "TXNDC5", "SUMF2", "CA13", "ADGRG2", "LRP11", "CCDC80",
        "CHRDL2", "ENPP7", "MEP1B", "KYAT1", "CLUL1", "NOMO1", "SEMA3F", "TYRO3", "IGFBPL1",
        "CD164", "FAM3C", "LRIG1", "SIGLEC7", "PILRB", "ADGRE2", "DPP7", "GHRL", "CLEC5A",
        "PAG1", "DIABLO", "NECTIN2", "NPDC1", "CDHR5", "CLMP", "CLSTN2", "TINAGL1", "RTN4R",
        "REG4", "SOST", "FCRL1", "ACP6", "CD2AP", "METRNL"
    ],
    "Target 96 Development": [
        "CCN5", "CTSF", "MATN2", "HTRA2", "MFGE8", "NUDT5", "ACAN", "ITGB1", "DKK3", "SPINT2",
        "APP", "HAVCR2", "DSC2", "TMSB10", "ANGPTL4", "INHBC", "SPINK1", "CLEC11A", "PDGFRB", "COCH",
        "FCRL5", "PILRA", "B4GALT1", "CD300LG", "PARK7", "BLVRB", "PLA2G4A", "CD74", "CCL21", "PTPRF",
        "BCAM", "SIRPB1", "IGF2R", "P4HB", "FUCA1", "ESAM", "ENPP2", "BST1", "CST6", "MYOC",
        "SNAP29", "WFIKKN2", "CDON", "CD177", "NID2", "DAG1", "ADGRE5", "PTPN6", "CD109", "VSIG4",
        "CRIM1", "CGA", "CLEC14A", "CA6", "MIF", "CRELD2", "ITGA5", "PEAR1", "MESD", "CD99L2",
        "SCARF1", "LGMN", "SEMA7A", "COLEC12", "GUSB", "FUT3_FUT5", "STIP1", "B4GAT1", "CD69", "CRHBP",
        "MSMB", "SPINT1", "FSTL3", "IL13RA1", "CD58", "CCN3", "CNTN4", "CA2", "XG", "ARSA",
        "PPIB", "SPINK5", "OMD", "PEBP1", "PAMR1", "ROBO1", "FCER2", "LAMA4", "LAIR1", "RELT",
        "TPP1", "CD209"
    ],
    "Target 96 Organ Damage": [
        "PTN", "NPPC", "PSMA1", "NOS3", "ALDH3A1", "BTC", "CAPG", "DPP6", "WAS", "CSNK1D", "PXN", "PGF", 
        "KIR3DL1", "RARRES1", "RASSF2", "FOSB", "METAP1", "BID", "MAX", "FKBP1B", "NUCB2", "RASA1", 
        "TMPRSS15", "PVALB", "CES2", "INPPL1", "EDIL3", "HPGDS", "ENTPD6", "PPM1B", "AIFM1", "LHB", 
        "EPO", "FES", "FGR", "NCF2", "CALR", "GALNT10", "RRM2B", "MAEA", "PON2", "PTK7", "FOXO1", 
        "FABP9", "MVK", "PLXDC1", "ENAH", "MAGED1", "NUB1", "PRKAB1", "CA14", "CLEC1A", "SIRT5", 
        "TIGAR", "LAT2", "ERBIN", "PDP1", "PTPRJ", "TNNI3", "ITGB1BP1", "CA12", "PLIN1", "NBN", 
        "PRKRA", "ATP6AP2", "AGR2", "CALCA", "CRH", "YES1", "LTA4H", "STXBP3", "SERPINA9", "DSG4", 
        "VASH1", "SMAD1", "PDCD1", "BAMBI", "ST3GAL1", "LRP1", "TOP2B", "CNTN2", "BANK1", "ENTPD2", 
        "MAP4K5", "STX8", "RCOR1", "EGFL7", "PDGFC", "CLSPN", "AMN", "HAVCR1", "ADGRG1"
    ],
    "Target 96 Cell Regulation": [
        "CFC1", "BCR", "ARSB", "CRISP2", "BGN", "OMG", "DNAJB1", "LGALS7_LGALS7B", "RGS8", "FLI1", 
        "ZBTB16", "ARHGAP1", "IRAG2", "DCTN2", "IQGAP2", "TP53", "NFKBIE", "STX16", "PCDH17", "BCL2L11", 
        "STX6", "HS6ST1", "ENTPD6", "VAMP5", "TCL1B", "GH2", "COL4A1", "PFKM", "DCBLD2", "KAZALD1", 
        "NINJ1", "ATG4A", "SORCS2", "SKAP1", "APBB1IP", "LRRN1", "METAP1D", "LYPD1", "CPXM1", 
        "HS3ST3B1", "WASF3", "DKKL1", "SLAMF8", "ARHGEF12", "PRDX6", "MAP2K6", "PROK1", "GCNT1", 
        "SULT2A1", "GALNT2", "ZBTB17", "TACSTD2", "GSAP", "TNFRSF10A", "GFRA2", "WNT9A", "CRX", 
        "SIGLEC6", "VEGFD", "IGSF3", "DDAH1", "NFATC1", "PAK4", "GCG", "JUN", "CBL", "SIGLEC10", "NCLN", 
        "WASF1", "AGR3", "CAMKK1", "AMIGO2", "TAFA5", "SEZ6L2", "CDNF", "MOG", "BOC", "TDRKH", "KLK12", 
        "OPTC", "CLSTN3", "LYAR", "GKN1", "PREB", "SLITRK2", "PODXL2", "MRC2", "FGF21", "IL17RB", 
        "SLITRK6", "SEMA4C", "TACC3"
    ],
    "Target 96 Mouse Exploratory": [
        "Acvrl1", "Adam23", "Ahr", "Apbb1ip", "Axin1", "Ca13", "Cant1", "Casp3", "Ccl2", "Ccl20", "Ccl3", 
        "Ccl5", "Cdh6", "Clmp", "Clstn2", "Cntn1", "Cntn4", "Crim1", "Csf2", "Ctgfa", "Cxcl1", "Cxcl9", 
        "Cyr61", "Dctn2", "Ddah1", "Dll1", "Dlk1", "Eda2r", "Eno2", "Epcam", "Epo", "Erbb4", "Fas", 
        "Fli1", "Fst", "Fstl3", "Foxo1", "Gcg", "Gdnf", "Gfra1", "Ghrl", "Hgf", "Igsf3", "Il10", 
        "Il17a", "Il17f", "Il1a", "Il1b", "Il23r", "Il5", "Il6", "Itgb1bp2", "Itgb6", "Kitlg", "Lgmn", 
        "Lpl", "Map2k6", "Matn2", "Mia", "Nadk", "Notch3", "Ntf3", "Pak4", "Parp1", "Pdgfb", "Pla2g4a", 
        "Plin1", "Plxna4", "Ppp1r2", "Prdx5", "Qdpr", "Rgma", "Riox2", "Sez6l2", "S100a4", "Snap29", 
        "Tgfbr3", "Tgfb1", "Tgfa", "Tnfrsf11b", "Tnfrsf12a", "Tnfsf12", "Tnni3", "Tnr", "Tnf", "Tpp1", 
        "Vegfd", "Vsig2", "Wfikkn2", "Wisp1", "Yes1"
    ]
}

# All unique biomarkers (no duplicates)
ALL_BIOMARKERS = sorted(set(sum(PANEL_MAP.values(), [])))

def import_mgh_covid_data(
    data_dir="../Data/MGH_Olink_COVID_Apr_27_2021/",
    na_values=None,
    return_var_desc=False,
    id_col="subject_id"
):
    """
    Import and preprocess MGH COVID clinical, OLINK NPX, and variable description data.
    Automatically finds and merges all relevant files for downstream analysis.
    NPX data is automatically pivoted to wide format (one row per subject, one column per assay).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data files (default: '../Data/MGH_Olink_COVID_Apr_27_2021/').
    na_values : list or str, optional
        Values to treat as missing (default: ['', 'NA', 'N/A']).
    return_var_desc : bool, optional
        Whether to return the variable description DataFrame.
    id_col : str, optional
        Name of the subject/sample ID column to use for merging (default: 'subject_id').

    Returns
    -------
    merged_df : pd.DataFrame
        Merged clinical + OLINK NPX data, index = subject_id.
    var_desc_df : pd.DataFrame, optional
        Variable descriptions, if requested and available.
    """
    if na_values is None:
        na_values = ['', 'NA', 'N/A']

    # --- Find files ---
    clinical_pattern = os.path.join(data_dir, '*Clinical_Info.txt')
    npx_pattern = os.path.join(data_dir, '*OLINK_NPX.txt')
    var_desc_pattern = os.path.join(data_dir, '*Variable_descriptions*.xlsx')

    clinical_files = glob.glob(clinical_pattern)
    npx_files = glob.glob(npx_pattern)
    var_desc_files = glob.glob(var_desc_pattern)

    if not clinical_files:
        raise FileNotFoundError(f"No clinical info file found in {data_dir}")
    if not npx_files:
        raise FileNotFoundError(f"No OLINK NPX file found in {data_dir}")
    clinical_file = clinical_files[0]
    npx_file = npx_files[0]
    var_desc_file = var_desc_files[0] if var_desc_files else None

    # --- Import clinical info ---
    clinical_df = pd.read_csv(
        clinical_file,
        sep=';',
        na_values=na_values,
        dtype=str
    )
    clinical_df.columns = clinical_df.columns.str.strip()
    for col in clinical_df.columns:
        try:
            clinical_df[col] = pd.to_numeric(clinical_df[col])
        except Exception:
            pass
    if 'subject_id' in clinical_df.columns:
        clinical_df['subject_id'] = clinical_df['subject_id'].astype(str).str.strip()
    else:
        raise ValueError("'subject_id' column not found in clinical info file.")

    # --- Import OLINK NPX data (long format) ---
    npx_df = pd.read_csv(
        npx_file,
        sep=';',  # Use semicolon delimiter
        na_values=na_values,
        dtype=str
    )
    npx_df.columns = npx_df.columns.str.strip()
    # Try to find the ID column (case-insensitive match)
    id_candidates = [c for c in npx_df.columns if c.lower() == id_col.lower()]
    if not id_candidates:
        print(f"ID column '{id_col}' not found in OLINK NPX file. Available columns:")
        print(npx_df.columns.tolist())
        raise ValueError(f"'{id_col}' column not found in OLINK NPX file.")
    npx_id_col = id_candidates[0]
    npx_df[npx_id_col] = npx_df[npx_id_col].astype(str).str.strip()
    # Pivot to wide format: subject_id as index, each Assay as a column, values=NPX
    if 'Assay' not in npx_df.columns or 'NPX' not in npx_df.columns:
        raise ValueError("NPX file must contain 'Assay' and 'NPX' columns for pivoting.")
    npx_wide = npx_df.pivot_table(
        index=npx_id_col,
        columns='Assay',
        values='NPX',
        aggfunc='first'  # or np.mean if there are duplicates
    )
    npx_wide.reset_index(inplace=True)
    npx_wide['subject_id'] = npx_wide[npx_id_col].astype(str).str.strip()
    # Convert numeric columns (except subject_id)
    for col in npx_wide.columns:
        if col in [npx_id_col, 'subject_id']:
            continue
        try:
            npx_wide[col] = pd.to_numeric(npx_wide[col])
        except Exception:
            pass

    # --- Merge clinical and NPX data using on='subject_id' ---
    merged_df = pd.merge(clinical_df, npx_wide, on='subject_id', how='inner')
    # Set index to subject_id
    merged_df.set_index('subject_id', inplace=True)
    # Do NOT drop the subject_id column (it's now the index)

    # --- Import variable descriptions ---
    var_desc_df = None
    if var_desc_file is not None:
        try:
            var_desc_df = pd.read_excel(var_desc_file)
            var_desc_df.columns = var_desc_df.columns.str.strip()
        except Exception as e:
            print(f"Warning: Could not read variable description file: {e}")
            var_desc_df = None

    if return_var_desc:
        return merged_df, var_desc_df
    else:
        return merged_df

def expand_biomarkers(user_biomarkers, panel_map=PANEL_MAP, all_biomarkers=ALL_BIOMARKERS):
    """
    Expand a list of user-specified biomarkers/panels into a unique list of biomarkers.
    """
    selected = set()
    for item in user_biomarkers:
        if item in panel_map:
            selected.update(panel_map[item])
        elif item in all_biomarkers:
            selected.add(item)
        else:
            raise ValueError(f"{item} is not a recognized panel or biomarker.")
    return sorted(selected)

def mgh_LDA(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    n_components=2,
    ax=None,
    legend_loc='best',
    title=None,
    cmap='tab10',
    alpha=0.8,
    s=40,
    vertical_jitter=True,
    group_labels=None,
    legend='inside'
):
    """
    Perform and plot LDA on MGH COVID data, coloring by any group/label column.
    Handles both binary and multiclass LDA.

    Parameters
    ----------
    data : pd.DataFrame
        Merged clinical + OLINK NPX data (from import_mgh_covid_data).
    label_col : str
        Column in data to use for coloring/groups (e.g., 'COVID', 'Acuity_max', etc.).
    feature_cols : list of str, optional
        Columns to use as features (default: all columns not in clinical or label_col).
    biomarkers : list of str, optional
        List of biomarker names and/or panel names to include in the analysis.
        If provided, only these biomarkers will be used. Panel names will be expanded to all biomarkers in that panel.
    panel_map : dict, optional
        Mapping from panel names to lists of biomarkers (required if biomarkers is used).
    all_biomarkers : list, optional
        List of all available biomarker names (required if biomarkers is used).
    n_components : int, optional
        Number of LDA components to compute/plot (default: 2).
    ax : matplotlib axis, optional
        Axis to plot on (default: creates new figure).
    legend_loc : str, optional
        Location of legend (default: 'best').
    title : str, optional
        Plot title (default: auto-generated).
    cmap : str, optional
        Matplotlib colormap for groups (default: 'tab10').
    alpha : float, optional
        Point transparency (default: 0.8).
    s : int, optional
        Point size (default: 40).
    vertical_jitter : bool, optional
        If True and n_components==1, add vertical jitter for 1D LDA plot (default: True).
    group_labels : dict, optional
        Dictionary mapping group values to custom legend labels. If not provided, uses group values as labels.
    legend : str, optional
        'inside' (default) to show legend inside plot, 'outside' to place legend outside plot area (right side).

    Returns
    -------
    lda : LinearDiscriminantAnalysis
        Fitted LDA object.
    X_lda : np.ndarray
        LDA-transformed data.
    """
    # --- SMART FEATURE SELECTION ---
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    elif feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Warning: The following biomarkers are missing from the data and will be skipped: {missing_features}")
        feature_cols = [col for col in feature_cols if col in data.columns]
        if not feature_cols:
            raise ValueError("None of the requested biomarkers/features are present in the data.")
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Warning: The following biomarkers are missing from the data and will be skipped: {missing_features}")
        feature_cols = [col for col in feature_cols if col in data.columns]
        if not feature_cols:
            raise ValueError("None of the requested biomarkers/features are present in the data.")

    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values

    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    explained_var = lda.explained_variance_ratio_ * 100  # percent variance explained

    # Plot (unchanged)
    is_1d = X_lda.shape[1] == 1
    group_sep = 1.0  # default group separation
    if ax is None:
        if legend == 'outside':
            fig_width = 12
        else:
            fig_width = 8
        if is_1d and not vertical_jitter:
            fig, ax = plt.subplots(figsize=(fig_width, 2))
        else:
            fig, ax = plt.subplots(figsize=(fig_width, 6))
    else:
        fig = None
    groups = np.unique(y)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(groups)))
    if is_1d:
        for i, group in enumerate(groups):
            mask = y == group
            if vertical_jitter:
                yvals = np.random.normal(i * group_sep, 0.04, size=np.sum(mask))
            else:
                yvals = np.zeros(np.sum(mask))
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_lda[mask, 0], yvals, label=label, color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        xlab = f"LDA 1 ({explained_var[0]:.1f}%)"
        ax.set_xlabel(xlab)
        if vertical_jitter:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
            ax.set_ylim(-0.25, group_sep * (len(groups)-1) + 0.25)
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_ylim(-0.1, 0.1)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position('zero')
    else:
        for i, group in enumerate(groups):
            mask = y == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=label, color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        xlab = f"LDA 1 ({explained_var[0]:.1f}%)"
        ylab = f"LDA 2 ({explained_var[1]:.1f}%)" if len(explained_var) > 1 else "LDA 2"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    if title is None:
        title = f"LDA: {label_col}"
    ax.set_title(title)
    if fig is not None:
        if legend == 'outside':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()
    return lda, X_lda

def mgh_LDA_varplot(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    max_components=None,
    ax=None,
    title=None,
    show_cumulative=True
):
    """
    Plot the explained variance ratio (scree plot) for LDA components.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import matplotlib.pyplot as plt
    import numpy as np

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values
    n_classes = len(np.unique(y))
    n_components = min(X.shape[1], n_classes - 1)
    if max_components is not None:
        n_components = min(n_components, max_components)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    explained_var = lda.explained_variance_ratio_ * 100  # percent

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    x = np.arange(1, len(explained_var) + 1)
    ax.bar(x, explained_var, color='tab:blue', alpha=0.8, label='Individual')
    if show_cumulative:
        ax.plot(x, np.cumsum(explained_var), color='tab:orange', marker='o', label='Cumulative')
    ax.set_xlabel('LDA Component')
    ax.set_ylabel('Explained Variance (%)')
    if title is None:
        title = f"{label_col} LDA Explained Variance"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return lda, explained_var

def mgh_LDA_loadings(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    n_components=2,
    component=0,
    sort=True,
    top_n=None,
    ax=None,
    title=None,
    color='tab:blue',
    highlight=None
):
    """
    Plot LDA loadings (coefficients) for each feature for a given component.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import matplotlib.pyplot as plt
    import numpy as np

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values
    n_classes = len(np.unique(y))
    n_components = min(X.shape[1], n_classes - 1, n_components)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    coefs = lda.coef_  # shape: (n_components, n_features) or (n_classes-1, n_features)
    # For multiclass, scikit-learn returns (n_classes-1, n_features)
    # For binary, shape is (1, n_features)
    if coefs.shape[0] < n_components:
        # If only one component, repeat for compatibility
        coefs = np.vstack([coefs] * n_components)
    # Select component
    loadings = coefs[component]
    features = np.array(feature_cols)
    if sort:
        idx = np.argsort(np.abs(loadings))[::-1]
        loadings = loadings[idx]
        features = features[idx]
    if top_n is not None:
        loadings = loadings[:top_n]
        features = features[:top_n]
    # Horizontal bar plot, largest at top
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(features))))
    else:
        fig = None
    ax.barh(features[::-1], loadings[::-1], color=color)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('LDA Loading (Coefficient)')
    ax.set_ylabel('Feature')
    if title is None:
        title = f'LDA Component {component+1} Top {len(features)} Loadings'
    ax.set_title(title)
    ax.set_yticklabels(features[::-1], fontsize=8)
    # Highlighting
    if highlight is not None:
        if isinstance(highlight, str):
            highlight = (highlight,)
        for label in ax.get_yticklabels():
            feat = label.get_text()
            if any(h in feat for h in highlight):
                label.set_color('red')
                label.set_fontweight('bold')
    plt.tight_layout()
    if fig is not None:
        plt.show()
    return lda, loadings, features

def mgh_LDA_biplot(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    n_components=2,
    top_n=10,
    scale_loadings=1.0,
    ax=None,
    title=None,
    color_map='tab10',
    standardize=True,
    group_labels=None,
    legend='inside'
):
    """
    LDA biplot: combine LDA scores (samples) and loadings (variables as vectors) in one plot.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.
    group_labels : dict, optional
        Dictionary mapping group values to custom legend labels. If not provided, uses group values as labels.
    legend : str, optional
        'inside' (default) to show legend inside plot, 'outside' to place legend outside plot area (right side).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import numpy as np

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X
    n_classes = len(np.unique(y))
    n_components = min(X.shape[1], n_classes - 1, n_components)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_proc, y)
    # Get loadings (coefficients)
    coefs = lda.coef_  # shape: (n_components, n_features) or (n_classes-1, n_features)
    if coefs.shape[0] < n_components:
        coefs = np.vstack([coefs] * n_components)
    features = np.array(feature_cols)
    # Select top_n features by max(abs(loading)) across available components
    if n_components == 1:
        loadings = coefs[0, :].reshape(1, -1)
        abs_max = np.abs(loadings[0])
        idx = np.argsort(abs_max)[::-1][:top_n]
        loadings_top = loadings[:, idx]
        features_top = features[idx]
        # Plot 1D biplot
        if ax is None:
            fig_width = 12 if legend == 'outside' else 8
            fig, ax = plt.subplots(figsize=(fig_width, 3))
        else:
            fig = None
        groups = np.unique(y)
        colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(groups)))
        for i, group in enumerate(groups):
            mask = y == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_lda[mask, 0], np.zeros(np.sum(mask)), label=label, color=colors[i], alpha=0.8, s=40, edgecolor='k', linewidth=0.5)
        # Overlay loadings as arrows along x-axis
        for i in range(top_n):
            ax.arrow(0, 0, loadings_top[0, i]*scale_loadings, 0,
                     color='red', alpha=0.8, head_width=0.05, head_length=0.08, length_includes_head=True)
            ax.text(loadings_top[0, i]*scale_loadings*1.15, 0.08,
                    features_top[i], color='red', fontsize=10, ha='center', va='bottom', fontweight='bold', rotation=45)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_xlabel('LDA 1')
        ax.set_yticks([])
        if title is None:
            title = f'LDA 1D Biplot: {label_col}'
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        if fig is not None:
            plt.show()
        return lda, X_lda, loadings, features
    else:
        # 2D biplot as before
        loadings = coefs[:2, :]  # shape: (2, n_features)
        abs_max = np.max(np.abs(loadings), axis=0)
        idx = np.argsort(abs_max)[::-1][:top_n]
        loadings_top = loadings[:, idx]
        features_top = features[idx]
        if ax is None:
            fig_width = 12 if legend == 'outside' else 8
            fig, ax = plt.subplots(figsize=(fig_width, 6))
        else:
            fig = None
        groups = np.unique(y)
        colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(groups)))
        for i, group in enumerate(groups):
            mask = y == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=label, color=colors[i], alpha=0.8, s=40, edgecolor='k', linewidth=0.5)
        # Overlay loadings as arrows
        for i in range(top_n):
            ax.arrow(0, 0, loadings_top[0, i]*scale_loadings, loadings_top[1, i]*scale_loadings,
                     color='red', alpha=0.8, head_width=0.08, head_length=0.12, length_includes_head=True)
            ax.text(loadings_top[0, i]*scale_loadings*1.15, loadings_top[1, i]*scale_loadings*1.15,
                    features_top[i], color='red', fontsize=10, ha='center', va='center', fontweight='bold')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_xlabel('LDA 1')
        ax.set_ylabel('LDA 2')
        if title is None:
            title = f'LDA Biplot: {label_col}'
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        if fig is not None:
            plt.show()
        return lda, X_lda, loadings, features

def mgh_PCA(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    n_components=2,
    ax=None,
    title=None,
    cmap='tab10',
    alpha=0.8,
    s=40,
    group_labels=None,
    legend='inside'
):
    """
    Perform and plot PCA on MGH COVID data, coloring by any group/label column.
    Handles both 1D and 2D PCA cases.

    Parameters
    ----------
    data : pd.DataFrame
        Merged clinical + OLINK NPX data (from import_mgh_covid_data).
    label_col : str
        Column in data to use for coloring/groups (e.g., 'COVID', 'Acuity_max', etc.).
    feature_cols : list of str, optional
        Columns to use as features (default: all columns not in clinical or label_col).
    biomarkers : list of str, optional
        List of biomarker names and/or panel names to include in the analysis.
        If provided, only these biomarkers will be used. Panel names will be expanded to all biomarkers in that panel.
    panel_map : dict, optional
        Mapping from panel names to lists of biomarkers (required if biomarkers is used).
    all_biomarkers : list, optional
        List of all available biomarker names (required if biomarkers is used).
    n_components : int, optional
        Number of PCA components to compute/plot (default: 2).
    ax : matplotlib axis, optional
        Axis to plot on (default: creates new figure).
    title : str, optional
        Plot title (default: auto-generated).
    cmap : str, optional
        Matplotlib colormap for groups (default: 'tab10').
    alpha : float, optional
        Point transparency (default: 0.8).
    s : int, optional
        Point size (default: 40).
    group_labels : dict, optional
        Dictionary mapping group values to custom legend labels. If not provided, uses group values as labels.
    legend : str, optional
        'inside' (default) to show legend inside plot, 'outside' to place legend outside plot area (right side).

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    X_pca : np.ndarray
        PCA-transformed data.
    """
    # --- SMART FEATURE SELECTION ---
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    elif feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]

    # --- Filter out missing biomarkers/features ---
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Warning: The following biomarkers are missing from the data and will be skipped: {missing_features}")
        feature_cols = [col for col in feature_cols if col in data.columns]
        if not feature_cols:
            raise ValueError("None of the requested biomarkers/features are present in the data.")

    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values

    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Fit PCA
    n_components = min(X.shape[1], n_components)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_var = pca.explained_variance_ratio_ * 100  # percent variance explained

    # Plot
    is_1d = X_pca.shape[1] == 1
    if ax is None:
        fig_width = 12 if legend == 'outside' else 8
        if is_1d:
            fig, ax = plt.subplots(figsize=(fig_width, 2))
        else:
            fig, ax = plt.subplots(figsize=(fig_width, 6))
    else:
        fig = None
    groups = np.unique(y)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(groups)))
    if is_1d:
        for i, group in enumerate(groups):
            mask = y == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_pca[mask, 0], np.zeros(np.sum(mask)), label=label, color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(f"PC 1 ({explained_var[0]:.1f}%)")
        ax.set_yticks([])
        ax.set_ylabel("")
    else:
        for i, group in enumerate(groups):
            mask = y == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(f"PC 1 ({explained_var[0]:.1f}%)")
        ax.set_ylabel(f"PC 2 ({explained_var[1]:.1f}%)" if len(explained_var) > 1 else "PC 2")
    if title is None:
        title = f"PCA: {label_col}"
    ax.set_title(title)
    if fig is not None:
        if legend == 'outside':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
        plt.tight_layout()
        plt.show()
    return pca, X_pca

def mgh_PCA_varplot(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    max_components=None,
    ax=None,
    title=None,
    show_cumulative=True
):
    """
    Plot the explained variance ratio (scree plot) for PCA components.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.

    Parameters
    ----------
    data : pd.DataFrame
        Merged clinical + OLINK NPX data (from import_mgh_covid_data).
    label_col : str
        Column in data to use for excluding from features (not used in PCA).
    feature_cols : list of str, optional
        Columns to use as features (default: all columns not in clinical or label_col).
    biomarkers : list of str, optional
        List of biomarker names and/or panel names to include in the analysis.
        If provided, only these biomarkers will be used. Panel names will be expanded to all biomarkers in that panel.
    panel_map : dict, optional
        Mapping from panel names to lists of biomarkers (required if biomarkers is used).
    all_biomarkers : list, optional
        List of all available biomarker names (required if biomarkers is used).
    max_components : int, optional
        Maximum number of PCA components to plot (default: all).
    ax : matplotlib axis, optional
        Axis to plot on (default: creates new figure).
    title : str, optional
        Plot title (default: auto-generated).
    show_cumulative : bool, optional
        Whether to plot cumulative explained variance (default: True).

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    explained_var : np.ndarray
        Explained variance ratio (percent) for each component.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n_components = X.shape[1]
    if max_components is not None:
        n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    explained_var = pca.explained_variance_ratio_ * 100  # percent

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    x = np.arange(1, len(explained_var) + 1)
    ax.bar(x, explained_var, color='tab:blue', alpha=0.8, label='Individual')
    if show_cumulative:
        ax.plot(x, np.cumsum(explained_var), color='tab:orange', marker='o', label='Cumulative')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance (%)')
    if title is None:
        title = f"PCA Explained Variance"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return pca, explained_var

def mgh_PCA_varplot_cum(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    max_components=None,
    ax=None,
    title=None
):
    """
    Plot only the cumulative explained variance for PCA components.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.
    Parameters and returns are the same as mgh_PCA_varplot.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n_components = X.shape[1]
    if max_components is not None:
        n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    explained_var = pca.explained_variance_ratio_ * 100  # percent

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    x = np.arange(1, len(explained_var) + 1)
    ax.plot(x, np.cumsum(explained_var), color='tab:orange', marker='o', label='Cumulative')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    if title is None:
        title = f"PCA Cumulative Explained Variance"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return pca, explained_var

def mgh_PCA_varplot_ind(
    data,
    label_col,
    feature_cols=None,
    biomarkers=None,
    panel_map=None,
    all_biomarkers=None,
    max_components=None,
    ax=None,
    title=None
):
    """
    Plot only the individual explained variance bars for PCA components.
    If 'biomarkers' is provided, only those biomarkers (and/or panels) are used.
    Parameters and returns are the same as mgh_PCA_varplot.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # SMART FEATURE SELECTION
    if biomarkers is not None:
        if panel_map is None:
            panel_map = PANEL_MAP
        if all_biomarkers is None:
            all_biomarkers = ALL_BIOMARKERS
        feature_cols = expand_biomarkers(biomarkers, panel_map, all_biomarkers)
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n_components = X.shape[1]
    if max_components is not None:
        n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    explained_var = pca.explained_variance_ratio_ * 100  # percent

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    x = np.arange(1, len(explained_var) + 1)
    ax.bar(x, explained_var, color='tab:blue', alpha=0.8, label='Individual')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance (%)')
    if title is None:
        title = f"PCA Individual Explained Variance"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return pca, explained_var
