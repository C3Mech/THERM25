#!/usr/bin/env python
# coding: utf-8

#########################################################################################
#   |===============================================================================|   #
#   |Therm23 - A Thermodynamic Property Estimator for Gas Phase Hydrocarbon Radicals|   #
#   |               and Molecules Based on Benson's Group Additivity Method         |   #
#   |===============================================================================|   #
#   |Copyright (c) 2022 by Sergio E. S. Martinez (sergioesmartinez@gmail.com)       |   # 
#   |           Updated by Pengzhi Wang from Comustion Chemistry Center             |   #
#   |           led by Henry J. Curran (henry.curran@universityofgalway.ie)         |   #
#   |                       Last updated: on April 2025                             |   #
#   |Permission is hereby granted, free of charge, to any person obtaining a        |   #
#   |copy of this software and associated documentation files (the 'Software'),     |   #
#   |to deal in the Software without restriction, including without limitation      |   #
#   |the rights to use, copy, modify, merge, publish, distribute, sublicense,       |   #
#   |and/or sell copies of the Software, and to permit persons to whom the          |   #
#   |Software is furnished to do so, subject to the following conditions:           |   #
#   |                                                                               |   #
#   |The above copyright notice and this permission notice shall be included in     |   #
#   |all copies or substantial portions of the Software.                            |   #
#   |                                                                               |   #
#   |THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     |   #
#   |IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       |   #
#   |FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    |   #
#   |AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         |   #
#   |LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING        |   #
#   |FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER            |   #
#   |DEALINGS IN THE SOFTWARE.                                                      |   #
#   |                                                                               |   #
#   |*******************************************************************************|   #
#   |Special Thanks to Kieran Patrick Somers for all the help. I'd like to thank    |   #
#   |Ultan Burke, Vaibhav patel, and Jennifer Power.                                |   #
#   |*******************************************************************************|   #
#########################################################################################

# **********************************************
# Python modules to work with
# **********************************************

import io
import os
import math
import sys
import glob
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate

from datetime import date
from scipy.optimize import curve_fit
from Core.Thermo import Thermo
from Core.Parsers import Parser
from Core.Parsers import CoeffReader
from Core.Inputs import Inputs
from Core.Fitts import WilhoitModel

psep = os.sep
R = 1.9872 # in cal units

def element_string(elem, number):
  number_str = str(number)
  if (elem == ""):
    raise Exception("elem must not be an empty string")
  if (number < 0):
    raise Exception("number = " + number_str +
                    " is not a reasonable number of elements")
  if (number == 0):
    return "    0"
  if (len(elem) + len(number_str) > 5):
    raise Exception("'" + elem + "'-'" + number_str +
                    "' is more than five characters long")
  fill = ' ' * (5 - len(elem) - len(number_str))
  return elem + fill + number_str


def generate_composition_str(elems):
  if (len(elems) > 5):
    raise Exception("Too many elements in " + str(elems))
  elem_5x5_str = [
      element_string(elem_str, number) for elem_str, number in elems.items()
  ]
  return "".join(elem_5x5_str)

def nasa_row1(BP, date):
  return 'G   300.000  5000.000{0:>9.3f}     1 ! Comment: Therm23, date: {1}\n'.format(BP,date)

def write_nasa_rows(f, name, elems, b, a, BP, date):
  name_len24 = f"{name[:24]:24}"
  f.write(name_len24 + generate_composition_str(elems) + nasa_row1(BP, date))
  f.write('{0:>15.8E}{1:>15.8E}{2:>15.8E}{3:>15.8E}{4:>15.8E}    2\n'.format(b[0], b[1], b[2], b[3], b[4]))
  f.write('{0:>15.8E}{1:>15.8E}{2:>15.8E}{3:>15.8E}{4:>15.8E}    3\n'.format(b[5], b[6], a[0], a[1], a[2]))
  f.write('{0:>15.8E}{1:>15.8E}{2:>15.8E}{3:>15.8E}                   4\n'.format(a[3], a[4], a[5], a[6]))
  
def f2s(val):
  return str(
      np.format_float_positional(np.float64(val), unique=False, precision=2))

def fixed_len(string, number, l):
  number_str = f2s(number)
  if (len(string) + len(number_str) >= l):
    answer = string + number_str
    print("#warning: '" + answer + "' is has at least " + str(l) +
          " characters. output might be ill-formatted.")
    return answer
  else:
    fill = ' ' * (l - len(string) - len(number_str))
    return string + fill + number_str


def print_elem_info(NumbC, NumbH, NumbO, NumbN):
  if (NumbC > 0):
    print("\t #C     = ", NumbC)
  if (NumbH > 0):
    print("\t #H     = ", NumbH)
  if (NumbO > 0):
    print("\t #O     = ", NumbO)
  if (NumbN > 0):
    print("\t #N     = ", NumbN)


class Outputs:
    """
    A class to represent the output data formats.
    This class takes different variables to convert it in strings or integers or float to save to dif. file formats;

    formats
    ------------------------------------------------------------------------------------------------------------------
    Re-run format: "*.rerun" is used to automatically re-run a specie or set of species' thermochemistry calculations.
    Doc format   : "*.doc" is used to save the basic parameters of each specie in a word format file.
    Therm format : "*.LST" is used to save the thermochemistry calculation based on GA's values such as Hf, S,
                            Cp300 K to 1500 K for further postprocessing.
    Dat format   : "*.dat" is used to save the NASA polynomial set's with a breaking point at 1000 K (Wilhoit
                        extrapolation is used here for the second set of polynomials)
    -------------------------------------------------------------------------------------------------------------------
    Attributes
    ----------------------
    f = Doc format file
    t = re-run format file
    q = LST format file
    z = dat format file

    """

    def __init__(self, DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, Radical, fGAVs,
                 Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500, RotoryApp,
                 SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity):
        """
            Constructs all the necessary attributes for the Outputs object.

            Parameters
            ---------------------------------------------------------------
            NamesFilesDoc   : str
                Doc document format
            NamesFilesTherm : str
                LST document format
            NamesFilesInp   : str
                Rerun document format
            ThermoFiles     : str
                Dat document format
            Speciename      : str
                Name of the specie working with
            Parent          : str
                A parent molecule
            Radical         : str
                Radical molecule based on parent molecule
            Formula         : str
                Chemical formula of the parent/radical molecule
            fGAVs           : list
                The Group Additivity Values (GAVs) gathered in a list
            Quantity        : int
                Number of times a GAVs must be used
            Enthalpy        : float (+/-)
                Standard enthalpy at 298 K in float format, positive or negative float
            Entropy         : float (+)
                Standard Entropy at 298 K in float format, positive float
            Cp300           : float (+)
                Heat capacity at 300 K in float format, positive float
            Cp400           : float (+)
                Heat capacity at 400 K in float format, positive float
            Cp500           : float (+)
                Heat capacity at 500 K in float format, positive float
            Cp600           : float (+)
                Heat capacity at 600 K in float format, positive float
            Cp800           : float (+)
                Heat capacity at 800 K in float format, positive float
            Cp1000          : float (+)
                Heat capacity at 1000 K in float format, positive float
            Cp1500          : float (+)
                Heat capacity at 1500 K in float format, positive float
            RotoryApp       : int
                Standard enthalpy at 298 K in float format, positive integer
            SymNumApp       : int
                Integer which represents the symmetry number of the specie, positive integer
            d2              : str
                A string giving the updated date
            NumberOfGAVs    : int
                Total number of GAVs used in fGAVs list in integer format, positive integer
            NumbC           : int
                Total number of carbon atoms in specie, positive integer
            NumbH           : int
                Total number of hydrogen atoms in specie, positive integer
            NumbO           : int
                Total number of oxygen atoms in specie, positive integer
            NumbN           : int
                Total number of nitrogen atoms in specie, positive integer
            CpINFlinear     : float
                Heat capacity at infinite temperature, positive float number
            linearity       : str
                A boolean variable, if YES it means molecule is linear other way (NO) is not.
        """
        self.NamesFilesDoc   = DocFileOut
        self.ThermoFiles     = DatFileOut
        self.NamesFilesTherm = LSTFileOut
        self.NamesFilesInp   = RerunFileOut
        self.Speciename      = Speciename
        self.Parent          = Parent
        self.Radical         = Radical
        self.Formula         = Formula
        self.fGAVs           = fGAVs
        self.Quantity        = Quantity
        self.Enthalpy        = Enthalpy
        self.Entropy         = Entropy
        self.Cp300           = Cp300
        self.Cp400           = Cp400
        self.Cp500           = Cp500
        self.Cp600           = Cp600
        self.Cp800           = Cp800
        self.Cp1000          = Cp1000
        self.Cp1500          = Cp1500
        self.RotoryApp       = RotoryApp
        self.SymNumApp       = SymNumApp
        self.d2              = d2
        self.NumberOfGAVs    = NumberOfGAVs
        self.NumbC           = NumbC
        self.NumbH           = NumbH
        self.NumbO           = NumbO
        self.NumbN           = NumbN
        self.CpINFlinear     = CpINFlinear
        self.linearity       = linearity.upper()

    def doc_format(self):
        """
            Constructs based on parameters provided to the Outputs object a *.doc format file with all the
            important thermochemistry properties summarise

            format example
            ==============================================================================================
            ----------------------------------------------------------------------
            SpecieName:	ETHANE
            Formula:	C2H6
            Units:	K,kcal
            	    Gr#	Group ID	Quantity
            	    1  	C/C/H3  	2
             Hf     S     CP300 CP400 CP500 CP600 CP800 CP1000 CP1500
            -20.02  54.84 12.44 15.48 18.48 21.24 25.68 29.18  34.70
            Number of rotors:	1
            Symmetry:	18
            Creation date:	12/07/2021
            Endspecies
            ----------------------------------------------------------------------
        """
        # **************************************************************************
        # BEGIN Therm23.date.doc
        # **************************************************************************
        #f.write("-" * 70)
        if self.Radical.lower() in ["y", "yes"]:
            #f.write("\nSpecieName and ParentName:\t" + str(self.Speciename) + "," + str(self.Parent) + "\n")
            f.write(" SPECIES\n " + str(self.Speciename) + "\nThermo estimation for radical\n " + str(self.Speciename) + "                                                    " + str(self.Formula) + "\n")
            f.write(" RADICAL BASED UPON PARENT " + str(self.Parent) + "\n PARENT FORMULA                         " + str(self.Parent) + "\nPARENT SYMMETRY\n")
        else:
            #f.write("\nSpecieName:\t" + str(self.Speciename) + "\n")
            f.write(" SPECIES\n " + str(self.Speciename) + "\nThermo estimation for molecule\n " + str(self.Speciename) + "                                                    " + str(self.Formula) + "\n")
        #f.write("Formula:\t" + str(self.Formula) + "\n")
        if self.linearity.lower() in ["y", "yes"]:
               f.write("UNITS:   " + "K,kcal" + "  -  LINEAR SPECIE\n")
        else:
               f.write("UNITS:   " + "K,kcal" + "  -  NONLINEAR SPECIE\n")       
        f.write("GROUPS   " + str(len(self.fGAVs)) + "\n")
        f.write("\tGr\t#\tGroup ID\tQuantity\n")
        for k in range(len(self.fGAVs)):
            if len(self.fGAVs[k]) < 4:
                f.write("\t  \t" + str(k + 1) + "\t-\t" + str(self.fGAVs[k]) + "  \t\t-\t" + str(self.Quantity[k]) + "\n")
            elif len(self.fGAVs[k]) >= 4 and len(self.fGAVs[k]) < 5:
                f.write("\t  \t" + str(k + 1) + "\t-\t" + str(self.fGAVs[k]) + "  \t-\t" + str(self.Quantity[k]) + "\n")
            elif  len(self.fGAVs[k]) >= 5 and  len(self.fGAVs[k]) < 7:
                f.write("\t  \t" + str(k + 1) + "\t-\t" + str(self.fGAVs[k]) + " \t-\t" + str(self.Quantity[k]) + "\n")
            else:
                f.write("\t  \t" + str(k + 1) + "\t-\t" + str(self.fGAVs[k]) + "\t-\t" + str(self.Quantity[k]) + "\n")

        f.write(" Hf    S     CP300 CP400 CP500 CP600 CP800 CP1000 CP1500\n")
        if self.Enthalpy < 0:
            if self.Enthalpy <= -10:
                f.write("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f \n" % ((self.Enthalpy),
                                                                              (self.Entropy), (self.Cp300),
                                                                              (self.Cp400), (self.Cp500), (self.Cp600),
                                                                              (self.Cp800), (self.Cp1000),
                                                                              (self.Cp1500)))

            else:
                f.write("%.2f  %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f \n" % ((self.Enthalpy),
                                                                               (self.Entropy), (self.Cp300),
                                                                               (self.Cp400), (self.Cp500), (self.Cp600),
                                                                               (self.Cp800), (self.Cp1000),
                                                                               (self.Cp1500)))

        else:
            if self.Enthalpy >= 10:
                f.write(" %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f \n" % ((self.Enthalpy),
                                                                               (self.Entropy), (self.Cp300),
                                                                               (self.Cp400), (self.Cp500), (self.Cp600),
                                                                               (self.Cp800), (self.Cp1000),
                                                                               (self.Cp1500)))
            else:
                f.write(" %.2f  %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f \n" % ((self.Enthalpy),
                                                                                (self.Entropy), (self.Cp300),
                                                                                (self.Cp400), (self.Cp500),
                                                                                (self.Cp600),
                                                                                (self.Cp800), (self.Cp1000),
                                                                                (self.Cp1500)))

        f.write("\t\tCPINF = " + str(self.CpINFlinear) + "\n")
        f.write("Number of rotors:\t" + str(self.RotoryApp) + "\n")
        f.write("Symmetry:\t" + str(self.SymNumApp) + "\n")
        if self.Radical.lower() in ["y", "yes"]:
           f.write("R ln(2) has been added to S to account\n")                                
           f.write("      for unpaired electron\n")                                          
           f.write("BOND DISS  \n")                                                         
        else:
           pass
        f.write("Creation date:\t" + str(self.d2) + "\n")
        f.write("ENDSPECIES\n")

        f.write("\n")
        f.write("\n")
        # ****************************************************************************
        # Therm23.date.doc END
        # ****************************************************************************
        return print("\t File " + (self.NamesFilesDoc).split(psep)[-1] + " updated.")

    def get_element_dict(self):
      elems = OrderedDict()
      elems["C"] = self.NumbC
      elems["H"] = self.NumbH
      elems["O"] = self.NumbO
      elems["N"] = self.NumbN
      return elems

    def therm_format(self):
        """
            Constructs based on parameters provided to the Outputs object a *.therm format file with all the
            important thermochemistry properties

            format example:
            =================================================================================
            SPECIES HF(298) S(298) CP300 CP400 CP500 CP600 CP800 CP1000 CP1500 DATE               ELEMENTS
            C2H6   -20.02   19.53  12.05 19.30 28.63 32.33 38.45 42.89  56.99  20/07/2021 Therm23 C 2 H 6 O 0 G 2
        """
        # ****************************************************************************
        # BEGIN Therm23.date.therm
        # ****************************************************************************
        # q.write("%s\t\t\t%.2f \t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\tTherm25\n" % (str(self.Formula),
        #                                                                                          (self.Enthalpy),
        #                                                                                          (self.Entropy),
        #                                                                                          (self.Cp300),
        #                                                                                          (self.Cp400),
        #                                                                                          (self.Cp500),
        #                                                                                          (self.Cp600),
        #                                                                                          (self.Cp800),
        #                                                                                          (self.Cp1000),
        #                                                                                          (self.Cp1500),
        #                                                                                          str(self.d2)))
        # q.write("SPECIES                   Hf     S     CP   300     400     500     600     800     1000     1500     DATE     ELEMENTS\n")        
        #
        if     abs(self.Enthalpy) == 0:
               self.Enthalpy  = abs(self.Enthalpy)
        elif   (self.Enthalpy)    == -0.00:
               self.Enthalpy  = abs(self.Enthalpy)
        elif   (self.Enthalpy)    == -0.0:
               self.Enthalpy  = abs(self.Enthalpy)
        elif   (self.Enthalpy)    == -0:
               self.Enthalpy  = abs(self.Enthalpy)
        # ==============================

        tab = "     \t"
        cp_str1 = f2s(self.Cp300) + tab + f2s(self.Cp400) + tab + f2s(self.Cp500) + tab
        cp_str2 = f2s(self.Cp600) + tab + f2s(self.Cp800) + tab + f2s(self.Cp1000) + tab + f2s(self.Cp1500) + tab
        l = 30
        output_str = (fixed_len(self.Speciename, self.Enthalpy, l) + tab 
                        + f2s(self.Entropy) + tab 
                        + cp_str1 
                        + cp_str2 
                        + str(self.d2) + tab + "        "
                        + str(self.NumbC) + tab + str(self.NumbH) + tab
                        + str(self.NumbO) + tab + str(self.NumbN) + tab 
                        + str(self.RotoryApp)) + "\n"
        q.write(output_str)

        # *****************************************************************************
        # Therm23.date.therm END
        # *****************************************************************************
        return print("\t File " + (self.NamesFilesTherm).split(psep)[-1] + " updated.")

    def rerun_format(self):
        """
            Constructs based on parameters provided to the Outputs object a *.rerun format file with all the
            important thermochemistry properties

            Parameters
            ----------

        """
        # ****************************************************************************
        # BEGIN Therm23.date.re
        # ****************************************************************************
        if len(self.fGAVs) > 1:
            ReRunGAVs = ",".join(self.fGAVs)
            ReRunQuan = ",".join(self.Quantity)
        else:
            ReRunGAVs = self.fGAVs[0]
            ReRunQuan = self.Quantity[0]
        if self.Radical.lower() in ["y", "yes"]:
            t.write(str(self.Speciename) + "\t" + str(self.Formula) + "\t" + str(self.NumberOfGAVs) + "\t" + 
                    str(ReRunGAVs) + "\t" + str(ReRunQuan) + "\t" + str(self.SymNumApp) + "\t" + str(self.RotoryApp) + "\t" +
                    str(self.Radical) + "\t" + str(self.Parent) + "\t" + str(self.NumbC) + "\t" + str(self.NumbH) + "\t" + 
                    str(self.NumbO) + "\t" + str(self.NumbN) + "\t" + str(self.linearity) + "\n")
        else:
            t.write(str(self.Speciename) + "\t" + str(self.Formula) + "\t" + str(self.NumberOfGAVs) + "\t" + 
                    str(ReRunGAVs) + "\t" + str(ReRunQuan) + "\t" + str(self.SymNumApp) + "\t" + str(self.RotoryApp) + "\t" + 
                    str(self.Radical) + "\t" + str(self.Parent) + "\t" + str(self.NumbC) + "\t" + str(self.NumbH) + "\t" + 
                    str(self.NumbO) + "\t" + str(self.NumbN) + "\t" + str(self.linearity) +  "\n")
        # *******************************************************************
        # *****************************************************************************
        # Therm23.date.rerun END
        # *****************************************************************************
        return print("\t File " + (self.NamesFilesInp).split(psep)[-1] + " updated.")

    def dat_format(self, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP):
        """
        Constructs based on parameters provided to the Outputs object a *.dat format file with 2 sets of
        NASA polynomial coefficients with a breaking point on T = 1000 K

        format example:
        =================================================================
          300.00  1000.00  5000.00
        C3H8       3/30/21 THERMC   3H   8    0    0G   300.000  5000.000 1390.000     1
         9.15541310E+00 1.72574139E-02-5.85614868E-06 9.04190155E-10-5.22523772E-14    2
        -1.75762439E+04-2.77418510E+01 2.40878470E-01 3.39548599E-02-1.60930874E-05    3
         2.83480628E-09 2.78195172E-14-1.40362853E+04 2.16500800E+01                   4
        """
        # *****************************************************************************
        # *****************************************************************************
        # Start of DAT file to save NASA polynomials
		# Headers for thermo file 
        # **************************************************************************************
        # The next are two main loops for the NASA polynomials coefficient to be printed
        # out to the corresponding output file. 1st loop for Breakin Point (BP) >= 1000 (K)
        # and the 2nd one for Breakin Points (BP) < 1000 (K).
        # **************************************************************************************
        write_nasa_rows(z, str(self.Speciename), self.get_element_dict(),
                        [b1, b2, b3, b4, b5, b6, b7], 
                        [a1, a2, a3, a4, a5, a6, a7], BP, self.d2)
           # ========================================================
           # UNZIPPING GROUP VALUES FOR SAVE THEM IN WORD FILE...
           # ========================================================
        # *****************************************************************************
        # Therm23.date.dat END
        # *****************************************************************************
        return print("\t File " + (self.ThermoFiles).split(psep)[-1] + " updated.")

    def ExtraWilhoit(self, a1, a2, a3, a4, a5, a6, a7):
		# **************************************************************************
		# definitions
        def funcCP(T, a1, a2, a3, a4, a5):
            # Cp/R   = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
            # CP/R here produce by code
            # CP here produce by code (updated)
            return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)*R
		
        def funcH(T, a6):
            T2 = T * T
            T4 = T2 * T2
            # H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
            # H here produce by code
            return (a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T2 * T / 4 + a5 * T4 / 5 + a6 / T) * R * T
		
        def funcS(T, a7):
            import math
            T2 = T * T
            T4 = T2 * T2
            # S/R  = a1 lnT + a2 T + a3 T^2 /2 + a4 T^3 /3 + a5 T^4 /4 + a7
            # S here produce by code
            return (a1 * np.log(T) + a2 * T + a3 * T2 / 2 + a4 * T2 * T / 3 + a5 * T4 / 4 + a7) * R

        def new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0):
            CPNewData = []
            for j in Tlimit:
                result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0, B=ValB0).getHeatCapacity(T=j)
                CPNewData.append(result2)
            ### ---------------------------------------------------------
            ### Getting Nasa polynomial coefficients for Cp(T)
            ### ----------------------------------------------------------
            def func(T, a1, a2, a3, a4, a5):
                return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)*R
        
            popt, pcov = curve_fit(func, Tlimit, CPNewData)
            b1 = (popt[0])
            b2 = (popt[1])
            b3 = (popt[2])
            b4 = (popt[3])
            b5 = (popt[4])
            #print(popt)
            #print("\t Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
            #print("\t %g  %g  %g  %g  %g " % (b1, b2, b3, b4, b5))
            fit1 = []
            for m in range(len(Tlimit)):
                fit1.append(func(Tlimit[m], b1, b2, b3, b4, b5))
            return Tlimit, fit1, b1, b2, b3, b4, b5
        def new_fit_optimzer2(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0):
            CPNewData = []
            for j in Tlimit:
                result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0, B=ValB0).getHeatCapacity2(T=j)
                CPNewData.append(result2)
            ### ---------------------------------------------------------
            ### Getting Nasa polynomial coefficients for Cp(T)
            ### ----------------------------------------------------------
            def func(T, a1, a2, a3, a4, a5):
                return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)*R
        
            popt, pcov = curve_fit(func, Tlimit, CPNewData)
            b1 = (popt[0])
            b2 = (popt[1])
            b3 = (popt[2])
            b4 = (popt[3])
            b5 = (popt[4])
            #print(popt)
            #print("\t Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
            #print("\t %g  %g  %g  %g  %g " % (b1, b2, b3, b4, b5))
            fit1 = []
            for m in range(len(Tlimit)):
                fit1.append(func(Tlimit[m], b1, b2, b3, b4, b5))
            return Tlimit, fit1, b1, b2, b3, b4, b5

        def new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5):
            CPNewData = []
            for j in Tlimit:
                result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0, B=ValB0).getEnthalpy(T=j)
                CPNewData.append(result2)
            ### ---------------------------------------------------------
            ### Getting Nasa polynomial coefficients for H
            ### ----------------------------------------------------------
            def funcH(T, b6):
                T2 = T * T
                T4 = T2 * T2
                # R = 0.0019872  # in kcal units
                # H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
                # H here produce by code
                return (b1 + b2 * T / 2 + b3 * T2 / 3 + b4 * T2 * T / 4 + b5 * T4 / 5 + b6 / T) * R * T / 1000 # in kcal units
        
            popt, pcov = curve_fit(funcH, Tlimit, CPNewData)
            b6 = (popt[0])
            #print(b6)
            #print(popt)
            #print("\t b1  b2  b3  b4  b5 ")
            #print("\t 2nd Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
            #print("\t %g  %g  %g  %g  %g %g " % (b1, b2, b3, b4, b5, b6))
            fit1 = []
            for m in range(len(Tlimit)):
                fit1.append(funcH(Tlimit[m], b6))
            return Tlimit, fit1, b1, b2, b3, b4, b5, b6
        
        def new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5):
            CPNewData = []
            for j in Tlimit:
                result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0,
                                       B=ValB0).getEntropy(T=j)
                CPNewData.append(result2)
            ### ---------------------------------------------------------
            ### Getting Nasa polynomial coefficients for S
            ### ----------------------------------------------------------
            def funcS(T, b7):
                T2 = T * T
                T4 = T2 * T2
                # S/R  = a1 lnT + a2 T + a3 T^2 /2 + a4 T^3 /3 + a5 T^4 /4 + a7
                # S here produce by code
                return (b1 * (np.log(T)) + b2 * T + b3 * T2 / 2 + b4 * T2 * T / 3 + b5 * T4 / 4 + b7) * R
                #return (b1 * (math.log(T)) + b2 * T + b3 * T2 / 2 + b4 * T2 * T / 3 + b5 * T4 / 4 + b7) * R

            popt, pcov = curve_fit(funcS, Tlimit, CPNewData)
            b7 = (popt[0])
            #print("\t 2nd Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
            #print("\t %g  %g  %g  %g  %g  %g" % (b1, b2, b3, b4, b5, b7))
            fit1 = []
            for m in range(len(Tlimit)):
                fit1.append(funcS(Tlimit[m], b7))
            #fit1=0
            #b1= b2= b3= b4= b5=b7=0
            return Tlimit, fit1, b1, b2, b3, b4, b5, b7
     
        # Solving for wilhoit
        NumberAtoms  = round((self.NumbC + self.NumbH + self.NumbO + self.NumbN), 2)
        Temperatures = [300, 400, 500, 600, 800, 1000, 1500]
        CpAll        = [self.Cp300, self.Cp400, self.Cp500, self.Cp600, self.Cp800, self.Cp1000, self.Cp1500]
        CpZERO       = (3.5) * R
        CpINF        = ((3 * (NumberAtoms) - 1.5) * R) / 2
        # 
        #plt.scatter( Temperatures, CpAll, marker="s", label='Cp exp. data', lw=4, color='black')
        #
        # *************************************
		# Wilhoit extrapolation section		
        # *************************************
        # Loop number 1 for Optimization of CP
        # *************************************
        #result = WilhoitModel().fitToData(Tlist=Temperatures, Cplist=CpAll, linear=self.linearity, nFreq=NumberAtoms, nRotors=self.RotoryApp, H298=self.Enthalpy, S298=self.Entropy, B0=500)
        # 
        result      = WilhoitModel().fitToData(Tlist=Temperatures, Cplist=CpAll, linear=self.linearity, nFreq=NumberAtoms, nRotors=self.RotoryApp, H298=self.Enthalpy, S298=self.Entropy, B0=500)
        ValsWilhoit = (str(result).split(","))
        ValCp0      = float((ValsWilhoit[0]).split("=")[-1])
        ValCpInf    = float((ValsWilhoit[1]).split("=")[-1])
        Vala0       = float((ValsWilhoit[2]).split("=")[-1])
        Vala1       = float((ValsWilhoit[3]).split("=")[-1])
        Vala2       = float((ValsWilhoit[4]).split("=")[-1])
        Vala3       = float((ValsWilhoit[5]).split("=")[-1])
        ValH0       = float((ValsWilhoit[6]).split("=")[-1])
        ValS0       = float((ValsWilhoit[7]).split("=")[-1])
        ValB0       = float((ValsWilhoit[8]).split("=")[-1])
        OrgB0       = int(ValB0)
        OrgH0       = int(ValH0)
        OrgS0       = int(ValS0)
        # 
        Tempo2000 = np.zeros(2001)  # np.zeros(471)
        Ti        = 1000
        DeltaT    = 1
        counter   = 0
        #
        for u in range(len(Tempo2000)):
            Tempo2000[counter] = (Ti + DeltaT * counter)
            counter += 1
        Tlimit  = Tempo2000

        Temps, newdata, b1, b2, b3, b4, b5        = new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
        # getting wilhoit enthalpies
        TempsH, newdataH, b1, b2, b3, b4, b5, b6 = new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, OrgH0, OrgS0, ValB0, b1, b2, b3, b4, b5)
        # getting wilhoit entropies
        TempsS, newdataS, b1, b2, b3, b4, b5, b7 = new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5)
        #print(b1, b2, b3, b4, b5, b7)
        # heat capacity definition: 
        def funcCP2(T, b1, b2, b3, b4, b5):
            #return (b1 + b2 * (T) + b3 * (T) ** 2 + b4 * (T) ** 3 + b5 * (T) ** 4)*(T**(1/2))
            return (b1 + b2 * (T) + b3 * (T) ** 2 + b4 * (T) ** 3 + b5 * (T) ** 4)*R

        Temp1000  = np.where(Temps  == 1000)

        Limit1    = (newdata[Temp1000[0][0]])
        Cp1000K   = funcCP(1000, a1, a2, a3, a4, a5)
        newdata   = [(x+round(Cp1000K-Limit1,3)) for x in newdata]
        ##
        #poptx, pcovx = curve_fit(funcCP2, FinalTempx, NewDataCP)
        #x1 = (poptx[0])
        #x2 = (poptx[1])
        #x3 = (poptx[2])
        #x4 = (poptx[3])
        #x5 = (poptx[4])
        #
        # ......................................................................................
        # ......................................................................................
        #fit0x  = []
        #for y in range(len(FinalTempx)):
        #    fit0x.append(funcCP2(FinalTemp[y], x1, x2, x3, x4, x5))
        #
        # ------------------------------------------------------------------------------
        # extra parameters section
        # ------------------------------------------------------------------------------
        #t1 = b1
        #t2 = b2
        #t3 = b3
        #t4 = b4
        #t5 = b5
        #t6 = b6
        #t7 = b7
        #print("\t Wilhoit guess CP: ")
        #print(newdata[0:10])
        # -------------------------------------------------------------------------------------------------
        # .........................................................................
        # Beginning Breaking Point Loop
        # .........................................................................
        # The main loop here try to guess an interception point to define
        # the Breaking Point (BP), in case this guessing fails, this loop
        # will fix the BP = 1000 (K).
        # .........................................................................
        # .........................................................................
        # LT fit section with derivatives and tangential point and line equations
        # .........................................................................
        TempsLT = np.zeros(2001)
        Ti = 300
        DeltaT = 1
        counter = 0
        for u in range(len(TempsLT)):
            TempsLT[counter] = (Ti + DeltaT * counter)
            counter += 1
        TiLT  = TempsLT
        CPLT = []
        for h in range(len(TiLT)):
            CPLT.append(funcCP(TiLT[h], a1, a2, a3, a4, a5))
        #print(TiLT)
        # Step 1 and 2:
        tck = interpolate.splrep(TiLT,CPLT)
        # .........................................................................
        # HT fit section with derivatives and tangential point and line equations
        # .........................................................................
        TempsHT = np.zeros(2001)
        Ti = 300
        DeltaT = 1
        counter = 0
        for u in range(len(TempsHT)):
            TempsHT[counter] = (Ti + DeltaT * counter)
            counter += 1
        TiHT  = TempsHT
        CPHT = []
        for h in range(len(TiHT)):
            CPHT.append(funcCP(TiHT[h], b1, b2, b3, b4, b5))
        # Step 1 and 2:
        tckHT = interpolate.splrep(TiHT,CPHT)
        #
        # .........................................................................
        # Comparing values of tnagential curves LT vs HT
        # .........................................................................
        # derivatives:
        dy1dx1 = interpolate.splev(TiLT,tck,  der=1)
        dy2dx2 = interpolate.splev(TiHT,tckHT,der=1)

        # definition to find the cross over point of the derivatives curves
        def interception(x1,y1,x2,y2):
           """
           Two given functions (raw-discrete data collected in list format from fuctions)
           f1 and f2 are treated, filter and manipulated in order to find out for those 
           coordinate points (x1,y1),(x2,y2) where both functions cross over each other.		   
           # Equations involved:
           # f1: y1 = m1(x1) => dy1 = m1(dx1) => m1 = dy1/dx1  
           # f2: y2 = m2(x2) => dy2 = m2(dx2) => m2 = dy2/dx2  
           # those cross over points must satisfy the next condition;
           # m1 = m2 => dy1/dx1 = dy2/dx2 => dy1*dx2 = dy2*dx1 => dy1*dx2 - dy2*dx1 = 0 
           # remember it may be multiple cross over points, this code will take always the 1st one only
           """
           Z   = []
           COP = []
           z = y1[0]*x2[0] - y2[0]*x1[0]
           if z > 0:
              for k in range(len(x1)):
                  z = y1[k]*x2[k] - y2[k]*x1[k]
                  if z > 0.005:
                     pass  
                  else:
                     Z.append(z)
                     COP.append(k)
                     break
           elif z < 0:
              for k in range(len(x1)):
                  z = y1[k]*x2[k] - y2[k]*x1[k]
                  if z < -0.005:
                     pass  
                  else:
                     Z.append(z)
                     COP.append(k)
                     break
           elif z == 0.000 :
                Z.append(z)
                COP.append(k)
           else:
                print("\t Something went wrong! Please contact develoeprs or try a different specie")
                sys.exit(1)
           return Z,COP
        
        Z, COP = interception(TiLT, dy1dx1, TiHT, dy2dx2)
        if len(COP) > 0:

           BP = int(COP[0])# + 500
           if BP < 3000 and BP >= 500: 
              Substration = 5000 - BP
              Tempo2000 = np.zeros(Substration+1)  # np.zeros(471)
              # Tempo5000[0] = 1000   #300
              #Ti = 1000
              Ti = BP
              #print(ValB0)
              DeltaT = 1
              counter = 0
              for u in range(len(Tempo2000)):
                  Tempo2000[counter] = (Ti + DeltaT * counter)
                  counter += 1
              Tlimit = Tempo2000
              #print(Tlimit)
              # getting wilhoit heat capacities
              Temps, newdata, b1, b2, b3, b4, b5       = new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
              # getting wilhoit enthalpies
              TempsH, newdataH, b1, b2, b3, b4, b5, b6 = new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, OrgH0, OrgS0, ValB0, b1, b2, b3, b4, b5)
              # getting wilhoit entropies
              TempsS, newdataS, b1, b2, b3, b4, b5, b7 = new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5)
              #print(b1, b2, b3, b4, b5, b7)
              # ------------------------------------------------------------------------------
              # extra parameters section
              # ------------------------------------------------------------------------------
              #t1 = b1
              #t2 = b2
              #t3 = b3
              #t4 = b4
              #t5 = b5
           elif BP < 500: 
              BP = 500 + BP
              Substration = 5000 - BP
              Tempo2000 = np.zeros(Substration+1)  # np.zeros(471)
              # Tempo5000[0] = 1000   #300
              #Ti = 1000
              Ti = BP
              #print(ValB0)
              DeltaT = 1
              counter = 0
              for u in range(len(Tempo2000)):
                  Tempo2000[counter] = (Ti + DeltaT * counter)
                  counter += 1
              Tlimit = Tempo2000
              #print(Tlimit)
              # getting wilhoit heat capacities
              Temps, newdata, b1, b2, b3, b4, b5       = new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
              # getting wilhoit enthalpies
              TempsH, newdataH, b1, b2, b3, b4, b5, b6 = new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, OrgH0, OrgS0, ValB0, b1, b2, b3, b4, b5)
              # getting wilhoit entropies
              TempsS, newdataS, b1, b2, b3, b4, b5, b7 = new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5)
              #print(b1, b2, b3, b4, b5, b7)
              # ------------------------------------------------------------------------------
              # extra parameters section
              # ------------------------------------------------------------------------------
              #t1 = b1
              #t2 = b2
              #t3 = b3
              #t4 = b4
              #t5 = b5
           else:
              Substration = 5000 - BP
              Tempo2000 = np.zeros(Substration+1)  # np.zeros(471)
              # Tempo5000[0] = 1000   #300
              #Ti = 1000
              Ti = BP
              #print(ValB0)
              DeltaT = 1
              counter = 0
              for u in range(len(Tempo2000)):
                  Tempo2000[counter] = (Ti + DeltaT * counter)
                  counter += 1
              Tlimit = Tempo2000

              # getting wilhoit heat capacities
              Temps, newdata, b1, b2, b3, b4, b5       = new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
              # getting wilhoit enthalpies
              TempsH, newdataH, b1, b2, b3, b4, b5, b6 = new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, OrgH0, OrgS0, ValB0, b1, b2, b3, b4, b5)
              # getting wilhoit entropies
              TempsS, newdataS, b1, b2, b3, b4, b5, b7 = new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5)
        else:
           BP = 1000
           Substration = 5000 - BP
           Tempo2000 = np.zeros(Substration)  # np.zeros(471)
           # Tempo5000[0] = 1000   #300
           #Ti = 1000
           Ti = BP
           #print(ValB0)
           DeltaT = 1
           counter = 0
           for u in range(len(Tempo2000)):
               Tempo2000[counter] = (Ti + DeltaT * counter)
               counter += 1
           Tlimit = Tempo2000
           #print(Tlimit)
           # getting wilhoit heat capacities
           Temps, newdata, b1, b2, b3, b4, b5       = new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
           # getting wilhoit enthalpies
           TempsH, newdataH, b1, b2, b3, b4, b5, b6 = new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, OrgH0, OrgS0, ValB0, b1, b2, b3, b4, b5)
           # getting wilhoit entropies
           TempsS, newdataS, b1, b2, b3, b4, b5, b7 = new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0, b1, b2, b3, b4, b5)
           print("Issue is here!!!")
        # End Breaking Point Loop
        # ------------------------------------------------------------------------------------------------------

        T1000K  = BP
        TH1000K = BP
        S1000K  = BP

        Cp1000K = funcCP(T1000K, a1, a2, a3, a4, a5)
        HT1000  = funcH(TH1000K, a6)
        ST1000K = funcS(S1000K, a7)

        # Some parameters for Cps
        Temp1000  = np.where(Temps  == BP)
        Temp1000S = np.where(TempsS == BP)
        Temp1000H = np.where(TempsH == BP)
        optimized = []
        Limit1    = (newdata[Temp1000[0][0]])
        Limit1H   = (newdataH[Temp1000H[0][0]])      
        Limit1S   = (newdataS[Temp1000S[0][0]])
        # --------------------------------------------------------
        # Equations:
        # Heat Capacity
        # --------------------------------------------------------
        # Beginning 'Heat capacity optimizer block'        
        # CP optimizer
        def funcCP2(T, b1, b2, b3, b4, b5):
            return (b1 + b2 * (T) + b3 * (T) ** 2 + b4 * (T) ** 3 + b5 * (T) ** 4)*R

        NewDataCP   = [(x+round(Cp1000K-Limit1,3)) for x in newdata]
        NewDataCP2  = NewDataCP #[(x+round(Cp1000K-Limit1,3)) for x in newdata]

        NewDataCPX  = []
        NewDataCPX2 = []

        for l in range(len(NewDataCP2)):
            if l == 0:
               pass #NewDataCPX.append(NewDataCP2[l])
            elif l > 0 and l < (len(NewDataCP2)-1):
               x0     = NewDataCP2[l-100]
               x1     = NewDataCP2[l]
               x2     = NewDataCP2[l+1]
               dx2dx1 = (x2 - x1)
               if   dx2dx1 >= 0.00000:
                    pass#NewDataCPX.append(l)
               elif dx2dx1  < 0.00000:
                    NewDataCPX.append(l)
            elif l == (len(NewDataCP2)-1):
                pass

        counter = 1
        if    len(NewDataCPX) > 0:
              Temps, newdata, b1, b2, b3, b4, b5 = new_fit_optimzer2(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0)
              T1000K  = BP
              TH1000K = BP
              S1000K  = BP
              Cp1000K = funcCP(T1000K, a1, a2, a3, a4, a5)
              HT1000  = funcH(TH1000K, a6)
              ST1000K = funcS(S1000K, a7)
              Temp1000  = np.where(Temps  == BP)
              Temp1000S = np.where(TempsS == BP)
              Temp1000H = np.where(TempsH == BP)
              optimized = []
              Limit1    = (newdata[Temp1000[0][0]])
              Limit1H   = (newdataH[Temp1000H[0][0]])      
              Limit1S   = (newdataS[Temp1000S[0][0]])
              NewDataCP   = [(x+round(Cp1000K-Limit1,3)) for x in newdata]
              NewDataCP2  = NewDataCP #[(x+round(Cp1000K-Limit1,3)) for x in newdata]
              popt, pcov  = curve_fit(funcCP2, Temps, NewDataCP)
              b1 = (popt[0])
              b2 = (popt[1])
              b3 = (popt[2])
              b4 = (popt[3])
              b5 = (popt[4])
              fit0  = []
              for m in range(len(Temps)):
                  fit0.append(funcCP2(Temps[m], b1, b2, b3, b4, b5))

        else:
              popt, pcov  = curve_fit(funcCP2, Temps, NewDataCP)
              b1 = (popt[0])
              b2 = (popt[1])
              b3 = (popt[2])
              b4 = (popt[3])
              b5 = (popt[4])
              fit0  = []
              for m in range(len(Temps)):
                  fit0.append(funcCP2(Temps[m], b1, b2, b3, b4, b5))

        print("\t","--" * 46)

        def funcH2(T, b6):
            T2 = T * T
            T4 = T2 * T2
            return (b1 + b2 * T / 2 + b3 * T2 / 3 + b4 * T2 * T / 4 + b5 * T4 / 5 + b6 / T) * R * T / 1000  # in kcal units
        def Coeffb6Opt(T, H1000K):
            """
            H(T)/Ro = a1*T + (a2/2)*T**2 + (a3/3)*T**3 + (a4/4)*T**4 + (a5/5)*T**5 + a6; solving with T = 298 K 
            """
            b6 = (H1000K / R / 1000) - b1 * T - (b2 / 2) * T ** 2 - (b3 / 3) * T ** 3 - (b4 / 4) * (T) ** 4 - (b5 / 5) * T ** 5
            return b6
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        NewDataH    = [(x+(HT1000-Limit1H)) for x in newdataH]
        popth, pcov = curve_fit(funcH2, Tlimit, NewDataH, method="lm")
        b6          = popth[0]
        # ......................................................................................
        fit1  = []
        b6opt = []
        for m in range(len(Tlimit)):
            fit1.append(funcH2(Tlimit[m], b6))
        AbsRelErrH = round(abs(((fit1[0]-HT1000)/HT1000)*100),2)
        b6 = Coeffb6Opt(T1000K, HT1000)
        # **************************************
        # End 'Enthalpy optimizer block'
        # **************************************
        # Beginning 'Entropy optimizer block'        
        # S optimizer
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        def funcS2(T, b7):
            T2 = T * T
            T4 = T2 * T2
            return (b1 * (np.log(T)) + b2 * T + b3 * T2 / 2 + b4 * T2 * T / 3 + b5 * T4 / 4 + b7) * R      
        def Coeffb7Opt(T, S1000K):
            """
            S(T)/Ro = a1*ln(T) + a2*T + (a3/2)*T**2 + (a4/3)*T**3 + (a5/4)*T**4 + a7;
            Natural logarithm calculated by 'np.log()'
            """
            b7 = (S1000K / R ) - b1 * np.log(T) - b2 * T - (b3 / 2) * T ** 2 - (b4 / 3) * T ** 3 - (b5 / 4) * T ** 4
            return b7
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        newdataS = [(x+(ST1000K-Limit1S)) for x in newdataS]
        popts, pcov = curve_fit(funcS2, Tlimit, newdataS, method="lm")
        b7 = (popts[0])
        # .................................................................................................
        fitS1  = []
        b7Sopt = []
        for m in range(len(Tlimit)):
            fitS1.append(funcS2(Tlimit[m], b7))
        AbsRelErrS = round(abs(((fitS1[0]-ST1000K)/ST1000K)*100),2)
        b7 = Coeffb7Opt(T1000K, ST1000K)
        # -------------------------------------------------------------------------------------------------
        # *************************************************************************************************
        # End 'Entropy optimizer block'
        # -------------------------------------------------------------------------------------------------
        return (b1, b2, b3, b4, b5, b6, b7, BP)
        #return (t1, t2, t3, t4, t5, t6, t7, BP)
        ################################################################################
# .............................................
# Definition(s) use(s) in this code
# .............................................



def new_fit_optimzer(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0):
    CPNewData = []
    for j in Tlimit:
        result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0,
                               B=ValB0).getHeatCapacity(T=j)
        CPNewData.append(result2)
    ### ---------------------------------------------------------
    ### Getting Nasa polynomial coefficients for Cp(T)
    ### ----------------------------------------------------------
    def func(T, a1, a2, a3, a4, a5):
        return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)*R

    popt, pcov = curve_fit(func, Tlimit, CPNewData)
    b1 = (popt[0])
    b2 = (popt[1])
    b3 = (popt[2])
    b4 = (popt[3])
    b5 = (popt[4])
    #print("\t b1  b2  b3  b4  b5 ")
    #print("\t Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
    #print("\t %g  %g  %g  %g  %g " % (b1, b2, b3, b4, b5))
    fit1 = []
    for m in range(len(Tlimit)):
        fit1.append(func(Tlimit[m], b1, b2, b3, b4, b5))
    return Tlimit, fit1, b1, b2, b3, b4, b5
################################################################################

def new_fit_optimzer_H(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0):
    CPNewData = []
    for j in Tlimit:
        result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0,
                               B=ValB0).getEnthalpy(T=j)
        CPNewData.append(result2)
    ### ---------------------------------------------------------
    ### Getting Nasa polynomial coefficients for H
    ### ----------------------------------------------------------
    def func(T, b1, b2, b3, b4, b5, b6):
        T2 = T * T
        T4 = T2 * T2
        # H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
        # H here produce by code
        return (b1 + b2 * T / 2 + b3 * T2 / 3 + b4 * T2 * T / 4 + b5 * T4 / 5 + b6 / T) * R * T / 1000 # in kcal units

    popt, pcov = curve_fit(func, Tlimit, CPNewData)
    b6 = (popt[0])
    #print("\t b1  b2  b3  b4  b5 ")
    #print("\t 2nd Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
    #print("\t %g  %g  %g  %g  %g %g " % (b1, b2, b3, b4, b5, b6))
    fit1 = []
    for m in range(len(Tlimit)):
        fit1.append(func(Tlimit[m], b1, b2, b3, b4, b5, b6))
    return Tlimit, fit1, b1, b2, b3, b4, b5, b6

##################################################################################

def new_fit_optimzer_S(Tlimit, ValCp0, ValCpInf, Vala0, Vala1, Vala2, Vala3, ValH0, ValS0, ValB0):#, b1, b2, b3, b4, b5):
    CPNewData = []
    for j in Tlimit:
        result2 = WilhoitModel(cp0=ValCp0, cpInf=ValCpInf, a0=Vala0, a1=Vala1, a2=Vala2, a3=Vala3, H0=ValH0, S0=ValS0,
                               B=ValB0).getEntropy(T=j)
        CPNewData.append(result2)
    ### ---------------------------------------------------------
    ### Getting Nasa polynomial coefficients for S
    ### ----------------------------------------------------------
    def funcS(T, b1, b2, b3, b4, b5, b7):
        T2 = T * T
        T4 = T2 * T2
        # S/R  = a1 lnT + a2 T + a3 T^2 /2 + a4 T^3 /3 + a5 T^4 /4 + a7
        # S here produce by code
        return (b1 * (np.log(T)) + b2 * T + b3 * T2 / 2 + b4 * T2 * T / 3 + b5 * T4 / 4 + b7) * R
        #return (b1 * (math.log(T)) + b2 * T + b3 * T2 / 2 + b4 * T2 * T / 3 + b5 * T4 / 4 + b7) * R
    #
    popt, pcov = curve_fit(funcS, Tlimit, CPNewData)
    b7 = (popt[0])
    #print("\t b1  b2  b3  b4  b5 b7")
    #print("\t 2nd Set of coefficients for High temperature fitting with breaking point at T = 1000 K")
    #print("\t %g  %g  %g  %g  %g  %g" % (b1, b2, b3, b4, b5, b7))
    fit1 = []
    for m in range(len(Tlimit)):
        fit1.append(funcS(Tlimit[m], b1, b2, b3, b4, b5, b7))
    return Tlimit, fit1, b1, b2, b3, b4, b5, b7


##################################################################################

def Variable2Test(POP):
    if POP.lower() in ["quit", "q", "end"]:
        "\t Not a proper choice, leaving now..."
        sys.exit(1)
    else:
        pass


##################################################################################
"""
	Some Units data format:
	
	Tdata  = ([300,400,500,600,800,1000,1500],'K'),
	Cpdata = ([6.948,6.948,6.949,6.954,6.995,7.095,7.493],'cal/(mol*K)'),
	H298   = (0,'kcal/mol'),
	S298   = (31.095,'cal/(mol*K)')
"""

# Definitions for user inputs:::


# ---------------------------------------
# End of Class definition
# ---------------------------------------

# ===============================================================================
# Main Code begin
# ================================================================================
"""
Author's note:
The first thing to fix when we are running our code is to set config up, which means
to let know to the python code if we wanna run in interactive or automatic mode the code.

This is done in this code as follows:
Python Therm23 code will ask the user one main question, 
> Do you want to run in Interactive/Automatic mode? "yes/no"
depends on the user answer the code will switch between the both modes available in this code,
but basically, the main difference between modes is that, in interactive mode you have to typoe 
everything stepo by step following the instrructions on screen while in automatic you have to type 
and saved previously everything in a *.inp file provided along with this code.

We hope you enjoy this code. SESM/2021.
"""

## RL: You want CWDPath based on how you use it below.
#      You want dir_path
## CWDPath = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
CWDPath = dir_path
today = date.today()
# dd/mm/YY
d1  = today.strftime("%d_%m_%Y")
d2  = today.strftime("%d/%m/%Y")
d22 = today.strftime("%d/%m/%Y")
print("\t","--" * 46)
print("\t |                                          Welcome                                         |")
print("\t |                                            to                                            |")
print("\t |                                          Therm23                                         |")
print("\t","--" * 46)
print("\t |        A Thermodynamic Property Estimator for Gas Phase Hydrocarbon Radicals and         |")
print("\t |                      Molecules Based on Benson's Group Additivity Method                 |")
print("\t","--" * 46)

basename = os.path.splitext(os.path.basename(__file__))[0]
## RL: This doesn't work for the current filename. It would better to manage to version number explicitly and elsewhere.
SplitVersion   = str(basename).split("_V")[-1]
print("\t |> Version of this software: " + str(SplitVersion) + " " * (62 - len(SplitVersion)) + "|")

print("\t |> Combustion Chemistry Centre (C\u00b3)                                                        |")
print("\t |> National University of Ireland Galway (NUIG), IE                                        |")
print("\t |> Sergio Martinez                                                                         |")
print("\t |> Prof. Henry Curran                                                                      |")
print("\t |>", d2, "                                                                             |")
print("\t |> s.martinez3@nuigalway.ie; sergioesmartinez@gmail.com                                    |")
print("\t |> henry.curran@nuigalway.ie                                                               |")
print("\t","--" * 46)

relativeGroup = "GroupsDir"
GroupDir = os.path.join(dir_path, relativeGroup)
GroupFiles = glob.glob(os.path.join(GroupDir, "*.grp"))
if len(GroupFiles) == 0:
    print("\t No GAV files were found!")
    print("\t Please check: '" + GroupDir + "' directory")
    print("\t and be sure that *.grp files are there before running the code again.")
    sys.exit(1)
else:
    pass

InChIFile = glob.glob(os.path.join(dir_path, "*data*.InChI"))
InputFile = glob.glob(os.path.join(dir_path, "*.inp"))
if len(InputFile) == 0:
    print("\t No input file was found!")
    print("\t Please check: '" + str(dir_path) + "' directory")
    print("\t and be sure that " + str(dir_path) + ".inp file is there before running the code again.")
    sys.exit(1)
else:
    pass

relativeOutput = "OutputsDir"
OUTPUT = os.path.join(dir_path, relativeOutput)
InputFileRe = glob.glob(os.path.join(OUTPUT, "*.rerun"))
THERMFiles  = glob.glob(os.path.join(OUTPUT, "*.dat"))
TMPinputs   = glob.glob(os.path.join(OUTPUT, "*.tmp"))
LSTinputs   = glob.glob(os.path.join(OUTPUT, "*.LST"))
Doc1        = glob.glob(os.path.join(OUTPUT, "*.doc"))
Doc2        = glob.glob(os.path.join(OUTPUT, "*.docx"))
DocRerun    = Doc1 + Doc2

TMPinputs2 = []
for y in TMPinputs:
    TMPinputs2.append(y.split(psep)[-1])

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)
    print("\t | Directory for output files:\t\t\t'" + relativeOutput + "' successfully created! |")
else:
    print("\t | Directory for output files:\t\t\t'" + relativeOutput + "' already exists! |")
#
print("\t | Reading GAVs from:\t\t\t\t'" + relativeGroup + "' directory folder |")
print("\t","**" * 46)
print("\t | There are " + str(len(GroupFiles)) + " file(s) that content GAVs in such directory folder                           |")
print("\t | The next is the list of file(s) with valid GAVs:                                         |")

for y in range(len(GroupFiles)):
    j = str(GroupFiles[y]).split(psep)[-1]
    if   y == 0:
         print("\t   " + j, end=' - ')
    elif y == (len(GroupFiles)-1):
         print(j)
    else:
         print(j, end=' - ')
print("\t","--" * 46)
#
# First and main question to user:
#
print("\t | Type 'END/end or quit/Q/q' to exit this program at any time                              |")
#
# Main menu:
# Choose between run a new thermochemistry or re-run an output file as input
#
print("\t",".." * 46)
print("\t | Main menu:                                                                               |")
print("\t | 1.- Interactive                                           (User needs to type all inputs)|")
print("\t | 2.- Automatic                    (The code uses " + str(dir_path) +".inp/any *.inp file as input)|")
print("\t | 3.- Re-calculate                (The code uses rerun/DOC files to recalculate properties)|")
print("\t | 4.- Thermo Plotter                 (H, S and Cp are plotted on single or multiple graphs)|")
print("\t | 5.- Thermo fitter                        (H, S and Cp are fitted and saved in a dat file)|")
#print("\t Exra mode:")
#print("\t 5.- CPs fitter\t\t    (Fits and plots data from LST files and extrapolates for H and S)")
# print("\t 4.- Plot thermochemistry properties (ΔH, ΔS and Cp). (Code uses OutputFileName.therm files)")
# print("\t 5.- Under dev")
print("\t",".." * 46)
#
MainMenu = Inputs("\t Type an option number:\t").main_switcher()
# Variable2Test(MainMenu)
# *************************************************************
# Switching to Interactive mode
# =============================================================
if   MainMenu == 1:
    #
    print("\t Calculating new thermochemistry in Interactive mode...")
    #
    Filename = input("\t Please provide the output file name:\n\t ")
    Filename = str(Filename).strip()
    Variable2Test(Filename)
    WordFile  = Filename + ".doc"
    DatFile   = Filename + ".dat"
    LSTFile   = Filename + ".LST"
    TmpFile   = Filename + ".tmp"
    RerunFile = Filename + ".rerun"
    # 
    for i in DocRerun:
        if WordFile == i.split(psep)[-1]:
           #print("\t i.split: ", i.split(psep)[-1])
           print("\t That file name already exist!")
           print("\t Would you like to overwrite it or to append new data to the file? (overwrite/w or 1 or append/a or 2)")
           AlreadyExFile = input("\t ") or "W"
           AlreadyExistingFileName = AlreadyExFile.upper()
        else:
           pass

    try:
       if   AlreadyExistingFileName in ["OVERWRITE", "W", 1]:
            print("\t Overwriting file...")
            #
            dfObj = Parser(GroupFiles).data()
            # *******************************
            # Switching to Interactive mode
            # *******************************    
            DocFileOut   = OUTPUT + psep + WordFile
            DatFileOut   = OUTPUT + psep + DatFile
            LSTFileOut   = OUTPUT + psep + LSTFile
            TmpFileOut   = OUTPUT + psep + TmpFile
            RerunFileOut = OUTPUT + psep + RerunFile
            # *************************************************************
            # Switching to Interactive mode
            # =============================================================
            # 
            f = open(DocFileOut, "w")
            z = open(DatFileOut, "w")
            z.write("THERMO\n")
            z.write("   300.00  1000.00  5000.00\n")
            q = open(LSTFileOut, "w")
            q.write("Units:\tK,calories\n")
            q.write("SPECIES                   \tHf        \tS     \tCP 300    \t400       \t500       \t600       \t800       \t1000      \t1500      \tDATE    \tELEMENTS  \tC   \tH   \tO   \tN   \tRotor \n")
            y = open(TmpFileOut, "w")
            y.write("DATE:\t" + str(d1) + "\n")
            y.write("GROUPNAME                 Hf     S     CP   300     400     500     600     800     1000     1500\n")
            t = open(RerunFileOut, "w")
            t.write("SpecieName          \tFormula          \tNumberOfGroups   \tGroupID   \tQuantity   \tSymmetryNumber   \tNumberOfRotors   \tIfRad   \tParentMol   \tC   \tH   \tO   \tN   \tLinearity\n")
            #
       elif AlreadyExistingFileName in ["APPEND", "A", 2]:
            print("\t Appending to file...")
            #
            #
            dfObj = Parser(GroupFiles).data()
            # *******************************
            # Switching to Interactive mode
            # *******************************    
            DocFileOut   = OUTPUT + psep + WordFile
            DatFileOut   = OUTPUT + psep + DatFile
            LSTFileOut   = OUTPUT + psep + LSTFile
            TmpFileOut   = OUTPUT + psep + TmpFile
            RerunFileOut = OUTPUT + psep + RerunFile
            # *************************************************************
            # Switching to Interactive mode
            # =============================================================
            z2 = open(DatFileOut, 'r+')  #pd.read_csv(DatFileOut)
            #with open(DatFileOut, 'r+') as z2:
            lines = z2.readlines()
            # move file pointer to the beginning of a file
            z2.seek(0)
            # truncate the file
            z2.truncate()
            # start writing lines except the last line
            # lines[:-1] from line 0 to the second last line
            z2.writelines(lines[:-1])
            z2.close()
            # 
            f = open(DocFileOut,   "a")
            z = open(DatFileOut,   "a")
            q = open(LSTFileOut,   "a")
            y = open(TmpFileOut,   "a")
            t = open(RerunFileOut, "a")
            #
    except:  
            #
            dfObj = Parser(GroupFiles).data()
            # *******************************
            # Switching to Interactive mode
            # *******************************    
            DocFileOut   = OUTPUT + psep + WordFile
            DatFileOut   = OUTPUT + psep + DatFile
            LSTFileOut   = OUTPUT + psep + LSTFile
            TmpFileOut   = OUTPUT + psep + TmpFile
            RerunFileOut = OUTPUT + psep + RerunFile
            # *************************************************************
            # Switching to Interactive mode
            # =============================================================
            # 
            f = open(DocFileOut, "w")
            z = open(DatFileOut, "w")
            z.write("THERMO\n")
            z.write("   300.00  1000.00  5000.00\n")
            q = open(LSTFileOut, "w")
            q.write("Units:\tK,calories\n")
            q.write("SPECIES                   \tHf        \tS     \tCP 300    \t400       \t500       \t600       \t800       \t1000      \t1500      \tDATE    \tELEMENTS  \tC   \tH   \tO   \tN   \tRotor \n")
            y = open(TmpFileOut, "w")
            y.write("DATE:\t" + str(d1) + "\n")
            y.write("GROUPNAME                 Hf     S     CP   300     400     500     600     800     1000     1500\n")
            t = open(RerunFileOut, "w")
            t.write("SpecieName          \tFormula          \tNumberOfGroups   \tGroupID   \tQuantity   \tSymmetryNumber   \tNumberOfRotors   \tIfRad   \tParentMol   \tC   \tH   \tO   \tN   \tLinearity\n")
            # 
    #
    try:
        #
        print("\t","==" * 46)
        print("\t |> Running in Interactive mode.")
        print("\t","==" * 46)
        for g in range(1000000):
            # ========================================================================
            # Block to provide specie name and chemical formula loop
            # ========================================================================
            Speciename = input("\t Type the species' name (molecular name):\t ")
            Variable2Test(Speciename)
            Speciename = str(Speciename).upper()
            while True:
                Formula = str(input("\t Type the chemical formula of " + Speciename + ":        \t ")).upper()
                Variable2Test(Formula)
                # Try to guess number of Carbon (C), Hydrogen (H) and Oxygen (O) atoms in specie
                if     "C" in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = 0
                     break
                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = 0
                     NumbN     = 0
                     break
                elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = 0
                     NumbO     = 0
                     NumbN     = 0
                     break
                elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                     NumbC     = 0
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = 0
                     NumbN     = 0
                     break
                elif   "C" not in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                     NumbC     = 0
                     NumbH     = 0
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = 0
                     break
                elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                     NumbC     = 0
                     NumbH     = 0
                     NumbO     = 0
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = 0
                     NumbN     = 0
                     break
                elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = 0
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = 0
                     break
                elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = 0
                     NumbO     = 0
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                     NumbC     = 0
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = 0
                     break
                elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                     NumbC     = 0
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = 0
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = 0
                     break
                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = 0
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" in Formula:
                     NumbC     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbH     = 0
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                     NumbC     = 0
                     NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                     break
                elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                     NumbC     = 0
                     NumbH     = 0
                     NumbO     = 0
                     NumbN     = 0
                     print("\t Not C, H, O or N atoms found in Chemical Formula provided by user.\n\t Try again typing a chemical formula this time.")
                     continue                 
            Linearity = input("\t Is this molecule linear?: (yes/y or no/n)\t ") 
            # ========================================================================
            # Block to provide specie name and chemical formula loop
            # ========================================================================
            print("\t","--"*46)
            Keeper2 = input("\t Is the data provided above correct?: (y/n)\t ") or "yes"
            if Keeper2.lower() in ["yes", "y"]:
                      print("\t Keeping data")
                      pass
            else:
                    # ========================================================================
                    # Block to provide specie name and chemical formula loop
                    # ========================================================================
                    Speciename = input("\t Type the specie's name (any string or formula):\t ")
                    Variable2Test(Speciename)
                    Speciename = str(Speciename).upper()
                    while True:
                        Formula = str(input("\t Type the chemical formula of " + Speciename + ":\t ")).upper()
                        Variable2Test(Formula)
                        # Try to guess number of Carbon (C), Hydrogen (H) and Oxygen (O) atoms in specie
                        if     "C" in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = 0
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = 0
                             NumbN     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = 0
                             NumbO     = 0
                             NumbN     = 0
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbC     = 0
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = 0
                             NumbN     = 0
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                             NumbC     = 0
                             NumbH     = 0
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = 0
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                             NumbC     = 0
                             NumbH     = 0
                             NumbO     = 0
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = 0
                             NumbN     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = 0
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = 0
                             NumbO     = 0
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbC     = 0
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = 0
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                             NumbC     = 0
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = 0
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = 0
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = 0
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" in Formula:
                             NumbC     = Inputs("\t Number of carbon  (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbH     = 0
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                             NumbC     = 0
                             NumbH     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbO     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbN     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                             NumbC     = 0
                             NumbH     = 0
                             NumbO     = 0
                             NumbN     = 0
                             print("\t Not C, H, O or N atoms found in Chemical Formula provided by user.\n\t Try again typing a chemical formula this time.")
                             continue                 
                    Linearity = input("\t Is this molecule linear?: (yes/y or no/n)\t") #Inputs
                    # ========================================================================
                    # Block to provide specie name and chemical formula loop
                    # ========================================================================
            print("\t","--"*46)
            CpINFlinear = round((3 * (NumbC + NumbH + NumbO) - 1.5) * (1.9), 2)
            # *************
            # INPUT GAVS
            # *************
            NumberOfGAVs = Inputs("\t Total number of groups in molecule?: (must be an integer) \t").input_number()
            fGAVs = []
            Quantity = []
            print("\t Give the group's name + Quantity")
            print("\t Type either C/C/H3 1   or  c/c/h3 1, or simply give the name and press enter; C/C/H3")
            #==============================
            # While loop to count the GAVs
            SumaGAVs = 0
            counter = 0
            while SumaGAVs < NumberOfGAVs:
                counter += 1
                NewGAV = input("\t " + str(counter) + " -  ")
                Variable2Test(NewGAV)
                UnGAV = str(str(NewGAV).split(" ")[0])
                UnQuanto = str(NewGAV).split(" ")[-1]
                if UnQuanto == UnGAV:
                    UnQuanto = "1"
                elif len(UnQuanto) == 0:
                    UnQuanto = "1"
                else:
                    pass
                fGAVs.append(UnGAV.upper())
                Quantity.append(UnQuanto)
                SumaGAVs = int(SumaGAVs) + int(UnQuanto)
                # End while loop
            # Extra loop allowing user to edit GAVs in case of misstyping
            print("\t","--"*46)
            Keeper = input("\t Are the groups provided above correct?: (y/n)\t ") or "yes"
            Keeper = Keeper.upper()
            if Keeper.lower() in ["yes", "y"]:
                      print("\t Keeping current groups")
                      pass
            else:
                      print("\t How many would you like to delete?")
                      PreDeleter = Inputs("\t ").input_number() or 0
                      for g in range(PreDeleter):
                          print("\t # of the GAV to delete:" )
                          Deleter = Inputs("\t ").input_number()
                          fGAVs.pop(Deleter-1)
                          NumberOfGAVs = NumberOfGAVs - int(Quantity[Deleter-1])
                          Quantity.pop(Deleter-1)
                          print("\t GAV removed")
                          print("\t New list is:")
                          if len(fGAVs) == 0:
                             print("\t List is empty now")
                          else:
                            for l in range(len(fGAVs)):
                                print("\t " + str(l + 1) + " - " + str(fGAVs[l]))
                      #==============================                          
                      # end of double checking loop
                      #==============================
                      # While loop to count the GAVs
                      NumberOfGAVs2 = Inputs("\t Give the number of additional groups: (must be an integer)\t").input_number()
                      NumberOfGAVs = int(NumberOfGAVs + NumberOfGAVs2)
                      SumaGAVs2 = 0
                      counter = 0
                      while SumaGAVs2 < NumberOfGAVs2:
                          counter += 1
                          NewGAV2 = input("\t " + str(counter) + " -  ")
                          Variable2Test(NewGAV2)
                          UnGAV2 = str(str(NewGAV2).split(" ")[0])
                          UnQuanto2 = str(NewGAV2).split(" ")[-1]
                          if UnQuanto2 == UnGAV2:
                              UnQuanto2 = "1"
                          elif len(UnQuanto2) == 0:
                              UnQuanto2 = "1"
                          else:
                              pass
                          fGAVs.append(UnGAV2.upper())
                          Quantity.append(UnQuanto2)
                          SumaGAVs2 = int(SumaGAVs2) + int(UnQuanto2)        
                      #==============================        
            print("\t","--"*46)
            SymNumApp = input("\t Total symmetry number:\t ")
            Variable2Test(SymNumApp)
            SymNumApp = int(SymNumApp)
            RotoryApp = int(input("\t Number of rotors:\t "))
            Variable2Test(RotoryApp)
            print("\t","**" * 46)
            print("\t Summarising")
            print("\t SpecieName:           {0}".format(Speciename))
            print("\t Formula:              {0}".format(Formula))
            print("\t Number of GAVs:       {0}".format(NumberOfGAVs))
            print("\t GAVs and Quantity: ")
            # --------------------
            GAVs = fGAVs
            for j in range(len(GAVs)):
                if   len(GAVs[j]) == 1:
                   print("\t "+str(j+1) +" - {0}               - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 2:
                   print("\t "+str(j+1) +" - {0}              - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 3:
                   print("\t "+str(j+1) +" - {0}             - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 4:
                   print("\t "+str(j+1) +" - {0}            - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 5:
                   print("\t "+str(j+1) +" - {0}           - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 6:
                   print("\t "+str(j+1) +" - {0}          - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 7:
                   print("\t "+str(j+1) +" - {0}         - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 8:
                   print("\t "+str(j+1) +" - {0}        - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 9:
                   print("\t "+str(j+1) +" - {0}       - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 10:
                   print("\t "+str(j+1) +" - {0}      - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 11:
                   print("\t "+str(j+1) +" - {0}     - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 12:
                   print("\t "+str(j+1) +" - {0}    - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 13:
                   print("\t "+str(j+1) +" - {0}   - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 14:
                   print("\t "+str(j+1) +" - {0}  - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 15:
                   print("\t "+str(j+1) +" - {0} - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 0:
                   print("\t WARNING! Error! Something went wrong! Not group found!")
                   sys.exit(1)
                else:
                   print("\t "+str(j+1) +" - {0} -\t{1}".format(GAVs[j], Quantity[j]))
            # --------------------
            print("\t Number of rotors:     {0}".format(RotoryApp))
            print("\t Symmetry number:      {0}".format(SymNumApp))
            print("\t","__" * 46)
            rads        = "No"
            Radical     = rads.upper()
            pparent     = "No"
            Parent      = pparent.upper()
            ##########

            if  Linearity.lower() in ["no","n"]:
                Cp0linear   = round((3.5) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
            elif Linearity.lower() in ["yes", "y"]:
                Cp0linear   = round((4.0) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 2.0) * (1.9), 2)

            (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                    speciesname=Speciename,
                                                                                    formula=Formula,
                                                                                    numberofgroups=NumberOfGAVs,
                                                                                    groupid=fGAVs,
                                                                                    quantity=Quantity,
                                                                                    symmetrynumber=SymNumApp,
                                                                                    numberofrotors=RotoryApp).thermo_props()
            # --------------------------------------------
            # Loop to check every GAV exist in GAV's files
            NewCorr_Test = []
            Corr_Test = list(ZIPTest)
            NarizList = []
            try:
                for p in range(len(Corr_Test)):
                    NarizList.append(Corr_Test[p][0])
            except:
                pass
            for h in range(len(fGAVs)):
                if fGAVs[h] in NarizList:
                    pass
                else:
                    NewCorr_Test.append(fGAVs[h])
            if len(NewCorr_Test) > 0:
                print("\t WARNING!!! Missed group(s)...")
                for j in range(len(NewCorr_Test)):
                    print("\t GAV (" + str(NewCorr_Test[j]) + ") not found!!!")
            else:
                pass
            # End of GAV's checker loops
            # ---------------------------
            print("\t Calculations:")
            print("\t Speciename        \tH(298K)\tS(298K)\tCp300K\t400K\t500K\t600K\t800K\t1000K\t1500K")
            # ==============================================================================================

            cp_str = ("\t" + str(Cp300) + "\t" + str(Cp400) + "\t" + str(Cp500) + "\t" + str(Cp600) 
                      + "\t" + str(Cp800) + "\t" + str(Cp1000) + "\t" + str(Cp1500))
            name_len18 = f"{Speciename[:18]:18}"

            print("\t " + name_len18 + "\t" + str(Enthalpy) + "\t" + str(Entropy) + cp_str)
            print("\t Cp0    = ", Cp0linear)
            print("\t CpINF  = ", CpINFlinear)

            Temps = [300, 400, 500, 600, 800, 1000, 1500]
            (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                  cp400=Cp400,
                                                  cp500=Cp500,
                                                  cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                  cp1500=Cp1500, temps=Temps).fit_termo()
	    
            (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, Radical,
                              fGAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                              RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)		
	    
	    
            # **************************************************************************
            # BEGIN Saving data
            # **************************************************************************
            print("\t Do you want to save this data to a file? (y/n)\t")
            SaveData = ((input("\t ")).strip()).upper()
            Variable2Test(SaveData)
            print("\t",".." * 46)

            if str(SaveData) == "YES" or str(SaveData) == "Y":
                Doc     = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, Radical, 
                              fGAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, 
                              Cp1500, RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).doc_format()
                ReRun   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, Radical,
                                fGAVs, Quantity,
                                Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500, RotoryApp,
                                SymNumApp,
                                d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).rerun_format()
                Thermoc = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, Radical,
                                  fGAVs,
                                  Quantity,
                                  Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500, RotoryApp,
                                  SymNumApp,
                                  d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).therm_format()
                Datac   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, "NO",
                                  fGAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                  RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP)

                # Saving used GAVs in specie and radicals
                Corr_TestR = Corr_Test
                for h in range(len(Corr_TestR)):
                    y.write(
                        str(Corr_TestR[h][0]) + "\t" + str(Corr_TestR[h][1]) + "\t" + str(Corr_TestR[h][2]) +
                        "\t" + str(Corr_TestR[h][3]) + "\t" + str(Corr_TestR[h][4]) + "\t" + str(
                            Corr_TestR[h][5]) +
                        "\t" + str(Corr_TestR[h][6]) + "\t" + str(Corr_TestR[h][7]) + "\t" + str(
                            Corr_TestR[h][8]) +
                        "\t" + str(Corr_TestR[h][9]) + "\n")
                # beginning of saving missed GAVs
                if len(NewCorr_Test) > 0:
                    y.write("# missing GAVs:\n")
                    for j in range(len(NewCorr_Test)):
                        y.write("# " + str(NewCorr_Test[j]) + "\n")
                else:
                    pass
                print("\t Please find your data in '\Therm23\OutputsDir\' directory\n")
                # end of saving missed GAVs
            elif str(SaveData).lower() in ["no", "n"]:
                print("\t Data was not saved.")
            else:
                print("\t Not a proper answer, not saving data and moving to next specie\n\t to exit program type 'END/end or quit/Q/q'.")
            # ****************************************************************
            #  1st loop for radical(s)/di-radical(s)
            # ****************************************************************
            print("\t Do you want to calculate a radical from this species? (y/n)")
            RadicalQ = Inputs("\t ").switcher2()
            RadicalQ = RadicalQ.upper()
            Variable2Test(RadicalQ)
            pparent = Speciename
            Parent = pparent.upper()
            # print(NumberOfGAVs)
            if RadicalQ.lower() in ["y", "yes"]:
                # ..........................................
                # ..........................................
                
                # ..........................................
                print("\t How many radicals do you want to calculate? (must be an integer)")
                RadicalsNum = Inputs("\t ").input_number()
                Variable2Test(RadicalsNum)
                for j in range(RadicalsNum):
                    RadfGAVs = fGAVs[:]
                    RadQuantity = Quantity[:]
                    RadNumberOfGAVs = NumberOfGAVs
                    rads = "yes"
                    Radical = rads.upper()
                    SpecienameRad = input("\t Type the name of radical #" + str(j+1) + " (radical mechanism name):\t ")
                    Variable2Test(SpecienameRad)
                    Speciename = str(SpecienameRad).upper()
                    # Try to guess number of Carbon (C), Hydrogen (H) and Oxygen (O) atoms in specie
                    # ========================================================================
                    # Block to provide specie name and chemical formula loop
                    # ========================================================================
                    while True:
                        Formula = input("\t Type the chemical formula of " + Speciename + ":\t ")
                        Variable2Test(Formula)
                        Formula = str(Formula).upper()
                        # Try to guess number of Carbon (C), Hydrogen (H) and Oxygen (O) atoms in specie
                        if     "C" in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = 0
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = 0
                             NumbNr     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = 0
                             NumbOr     = 0
                             NumbNr     = 0
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbCr     = 0
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = 0
                             NumbNr     = 0
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                             NumbCr     = 0
                             NumbHr     = 0
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = 0
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                             NumbCr     = 0
                             NumbHr     = 0
                             NumbOr     = 0
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = 0
                             NumbNr     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = 0
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = 0
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = 0
                             NumbOr     = 0
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbCr     = 0
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = 0
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                             NumbCr     = 0
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = 0
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = 0
                             break
                        elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = 0
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" in Formula:
                             NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbHr     = 0
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                             NumbCr     = 0
                             NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                             break
                        elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                             NumbCr     = 0
                             NumbHr     = 0
                             NumbOr     = 0
                             NumbNr     = 0
                             print("\t Not C, H, O or N atoms found in Chemical Formula provided by user.\n\t Try again typing a chemical formula this time.")
                             continue                 
                    Linearity = input("\t Is this molecule linear?:\t  (yes/no)\t") 
                    # ========================================================================
                    # Block to provide specie name and chemical formula loop
                    # ========================================================================
                    print("\t","--"*46)
                    Keeper2 = input("\t Is the data provided above correct?: (yes/y or no/n)\t ") or "yes"
                    if Keeper2.lower() in ["yes", "y"]:
                              print("\t Keeping data")
                              pass
                    else:
                            # ========================================================================
                            # Block to provide specie name and chemical formula loop
                            # ========================================================================
                            SpecienameRad = input("\t Type the radical's name (any string or formula):\t ")
                            Variable2Test(SpecienameRad)
                            Speciename = str(SpecienameRad).upper()
                            while True:
                                Formula = input("\t Type the chemical formula on " + Speciename + ":\t ")
                                Variable2Test(Formula)
                                Formula = str(Formula).upper()
                                # Try to guess number of Carbon (C), Hydrogen (H) and Oxygen (O) atoms in specie
                                if     "C" in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = 0
                                     break
                                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = 0
                                     NumbNr     = 0
                                     break
                                elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = 0
                                     NumbOr     = 0
                                     NumbNr     = 0
                                     break
                                elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                                     NumbCr     = 0
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = 0
                                     NumbNr     = 0
                                     break
                                elif   "C" not in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                                     NumbCr     = 0
                                     NumbHr     = 0
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = 0
                                     break
                                elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                                     NumbCr     = 0
                                     NumbHr     = 0
                                     NumbOr     = 0
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = 0
                                     NumbNr     = 0
                                     break
                                elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = 0
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = 0
                                     break
                                elif   "C" in Formula and "H" not in Formula and "O" not in Formula and "N" in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = 0
                                     NumbOr     = 0
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                                     NumbCr     = 0
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = 0
                                     break
                                elif   "C" not in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                                     NumbCr     = 0
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = 0
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" in Formula and "H" in Formula and "O" in Formula and "N" not in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = 0
                                     break
                                elif   "C" in Formula and "H" in Formula and "O" not in Formula and "N" in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = 0
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" in Formula and "H" not in Formula and "O" in Formula and "N" in Formula:
                                     NumbCr     = Inputs("\t Number of carbon   (C) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbHr     = 0
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     break
                                elif   "C" not in Formula and "H" in Formula and "O" in Formula and "N" in Formula:
                                     NumbCr     = 0
                                     NumbHr     = Inputs("\t Number of hydrogen (H) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbOr     = Inputs("\t Number of oxygen   (O) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     NumbNr     = Inputs("\t Number of nitrogen (N) atoms in " + Speciename + "? :\t ").input_number() #Inputs
                                     breakr
                                elif   "C" not in Formula and "H" not in Formula and "O" not in Formula and "N" not in Formula:
                                     NumbCr     = 0
                                     NumbHr     = 0
                                     NumbOr     = 0
                                     NumbNr     = 0
                                     print("\t Not C, H, O or N atoms found in Chemical Formula provided by user.\n\t Try again typing a chemical formula this time.")
                                     continue                 
                            Linearity = input("\t Is this molecule linear?: (yes/y or no/n)\t") #Inputs
                            # ========================================================================
                            # Block to provide specie name and chemical formula loop
                            # ========================================================================
                    print("\t","--"*46)
	    
                    # If guess fail, allow user to input manually the right values
	    
                    # we need to calculate the heat capacity at infinite temperature for further polynomial calculations
                    #CpINFlinearr = round((3 * (NumbCr + NumbHr + NumbOr) - 1.5) * (1.9), 2)
                    ###CpINFlinearr = round((3 * (NumbCr + NumbHr + NumbOr + NumbNr) - 2) * (1.9), 2)
                    if  Linearity.lower() in ["no", "n"]:
                        CpINFlinearr = round((3 * (NumbCr + NumbHr + NumbOr + NumbNr) - 1.5) * (1.9), 2)
                    elif Linearity.lower() in ["yes", "y"]:
                        CpINFlinearr = round((3 * (NumbCr + NumbHr + NumbOr + NumbNr) - 2.0) * (1.9), 2)
                    # *****************************************************************
                    print("\t","--" * 46)
                    print("\t These are the parent molecule's (" + str(Parent) + ") groups:")
                    for l in range(len(RadfGAVs)):
                        print("\t " + str(l+1) + " - " + str(RadfGAVs[l]))
                    print("\t ")
                    print("\t Keep these groups? (yes/no) - press Enter for default 'yes'")
                    print("\t if yes, you will be asked only for the BD group")
                    print("\t if you answered no, you will be asked for the number of groups to be deleted")
                    Keeper = input("\t ") or "yes"
                    Keeper = Keeper.upper()
                    if Keeper == "yes":
                        print("\t " + Keeper)
                    else:
                        pass
                    if Keeper.lower() in ["yes", "y"]:
                        print("\t Keeping the parent's groups")
                        print("\t Moving to the radicals")
                        print("\t","--" * 46)
                        pass
                    else:
                        print("\t How many would you like to delete?")
                        PreDeleter = Inputs("\t ").input_number()
                        for g in range(PreDeleter):
                            print("\t # of the GAV to delete:" )
                            Deleter = Inputs("\t ").input_number()
                            RadfGAVs.pop(Deleter-1)
                            RadNumberOfGAVs = RadNumberOfGAVs - int(RadQuantity[Deleter-1])
                            RadQuantity.pop(Deleter-1)
                            print("\t Parent's GAV removed.")
                            print("\t New list is:")
                            for l in range(len(RadfGAVs)):
                                print("\t " + str(l + 1) + " - " + str(RadfGAVs[l]))
                        print("\t Moving to the radicals")

                    NumberOfGAVsR = Inputs("\t Give the number of BD groups: (must be an integer)\t").input_number()
                    NumberOfGAVs2 = int(RadNumberOfGAVs + NumberOfGAVsR)
                    print("\t Enter BD type")
                    # ==============================
                    # Count the GAVs
                    SumaGAVs = 0
                    while SumaGAVs < NumberOfGAVsR:
                        NewGAV = input("\t ")
                        Variable2Test(NewGAV)
                        UnGAV = str(str(NewGAV).split(" ")[0])
                        UnQuanto = str(NewGAV).split(" ")[-1]
                        if UnQuanto == UnGAV:
                            UnQuanto = "1"
                        elif len(UnQuanto) == 0:
                            UnQuanto = "1"
                        else:
                            pass
                        RadfGAVs.append(UnGAV.upper())
                        RadQuantity.append(UnQuanto)
                        SumaGAVs = int(SumaGAVs) + int(UnQuanto)

                    SymNumApp = input("\t Total symmetry number:\t ")
                    Variable2Test(SymNumApp)
                    SymNumApp = int(SymNumApp)
                    RotoryApp = int(input("\t Number of rotors:\t "))
                    Variable2Test(RotoryApp)
                    # ***********************************************
                    print("\t","**" * 46)
                    print("\t Summarising")
                    print("\t RadicalName:          {0}".format(Speciename))
                    print("\t Formula:              {0}".format(Formula))
                    print("\t Number of GAVs:       {0}".format(NumberOfGAVs2))
                    print("\t GAVs and Quantity: ")
                    # --------------------
                    GAVs = RadfGAVs
                    for j in range(len(GAVs)):
                        if   len(GAVs[j]) == 1:
                           print("\t "+str(j+1) +" - {0}               - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 2:
                           print("\t "+str(j+1) +" - {0}              - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 3:
                           print("\t "+str(j+1) +" - {0}             - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 4:
                           print("\t "+str(j+1) +" - {0}            - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 5:
                           print("\t "+str(j+1) +" - {0}           - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 6:
                           print("\t "+str(j+1) +" - {0}          - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 7:
                           print("\t "+str(j+1) +" - {0}         - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 8:
                           print("\t "+str(j+1) +" - {0}        - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 9:
                           print("\t "+str(j+1) +" - {0}       - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 10:
                           print("\t "+str(j+1) +" - {0}      - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 11:
                           print("\t "+str(j+1) +" - {0}     - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 12:
                           print("\t "+str(j+1) +" - {0}    - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 13:
                           print("\t "+str(j+1) +" - {0}   - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 14:
                           print("\t "+str(j+1) +" - {0}  - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 15:
                           print("\t "+str(j+1) +" - {0} - {1}".format(GAVs[j], RadQuantity[j]))
                        elif len(GAVs[j]) == 0:
                           print("\t WARNING! Error! Something went wrong! Not group found!")
                           sys.exit(1)
                        else:
                           print("\t "+str(j+1) +" - {0} -\t{1}".format(GAVs[j], RadQuantity[j]))
                    # --------------------
                    print("\t Number of rotors:     {0}".format(RotoryApp))
                    print("\t Symmetry number:      {0}".format(SymNumApp))
                    print("\t","__" * 46)
                    # ***********************************************
                    # print(dfObj)
                    #Cp0linearRad = round((4.0) * (1.9), 2)
                    #CpINFlinearRad = round((3 * (NumbC + NumbH + NumbO) - 2.0) * (1.9), 2)
                    ##############
                    if  Linearity.lower() in ["no", "n"]:
                        Cp0linearRad   = round((3.5) * (1.9), 2)
                        CpINFlinearRad = round((3 * (NumbCr + NumbHr + NumbOr + NumbNr) - 1.5) * (1.9), 2)
                    elif Linearity.lower() in ["yes", "y"]:
                        Cp0linearRad   = round((4.0) * (1.9), 2)
                        CpINFlinearRad = round((3 * (NumbCr + NumbHr + NumbOr + NumbNr) - 2.0) * (1.9), 2)
                    ##############
                    (Enthalpy, Entropy, ZIPTestR, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                             speciesname=Speciename,
                                                                                                             formula=Formula,
                                                                                                             numberofgroups=NumberOfGAVs2,
                                                                                                             groupid=RadfGAVs,
                                                                                                             quantity=RadQuantity,
                                                                                                             symmetrynumber=SymNumApp,
                                                                                                             numberofrotors=RotoryApp).thermo_props_rads()
                    # -----------------------------------------------------------------
                    # Loop to check every GAV exist in GAV's files for radical species
                    NewCorr_Test2 = []
                    Corr_TestR = list(ZIPTestR)

                    NarizList2 = []
                    try:
                        for p in range(len(Corr_TestR)):
                            NarizList2.append(Corr_TestR[p][0])
                    except:
                        pass
                    for h in range(len(RadfGAVs)):
                        if RadfGAVs[h] in NarizList2:
                            pass
                        else:
                            NewCorr_Test2.append(RadfGAVs[h])
                    if len(NewCorr_Test2) > 0:
                        print("\t WARNING!!! Missed group(s)...")
                        for j in range(len(NewCorr_Test2)):
                            print("\t GAV (" + str(NewCorr_Test2[j]) + ") not found!!!")
                    else:
                        pass
                    # End of GAV's checker loops for radicals species
                    # -------------------------------------------------
                    print("\t",".." * 46)
                    # **********************
                    print("\t Calculations:")
                    print("\t Speciename        \tH(298K)\tS(298K)\tCp300K\t400K\t500K\t600K\t800K\t1000K\t1500K")

                    cp_str = ("\t" + str(Cp300) + "\t" + str(Cp400) + "\t" + str(Cp500) + "\t" + str(Cp600) 
                              + "\t" + str(Cp800) + "\t" + str(Cp1000) + "\t" + str(Cp1500))
                    name_len18 = f"{Speciename[:18]:18}"

                    print("\t " + name_len18 + "\t" + str(Enthalpy) + "\t" + str(Entropy) + cp_str)
                    print("\t Cp0    = ", Cp0linearRad)
                    print("\t CpINF  = ", CpINFlinearRad)
                    print("\t",".." * 46)
                    print("\t Do you want to save this data on file? (y/n)")
                    SaveData = input("\t ")
                    Variable2Test(SaveData)
	    
            
                    Temps = [300, 400, 500, 600, 800, 1000, 1500]

                    (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                          cp400=Cp400,
                                                          cp500=Cp500,
                                                          cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                          cp1500=Cp1500, temps=Temps).fit_termo()
            
            
                    (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, "yes",
                                      RadfGAVs, RadQuantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                      RotoryApp, SymNumApp, d2, NumberOfGAVs2, NumbCr, NumbHr, NumbOr, NumbNr, CpINFlinearr, Linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)		
            
            
            
            
            
                    # **************************************************************************
                    # BEGIN Saving data
                    # **************************************************************************
                    if str(SaveData).lower() in ["yes", "y"]:
                        Doc     = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent,
                                      Radical, RadfGAVs,
                                      RadQuantity,
                                      Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                      RotoryApp,
                                      SymNumApp,
                                      d2, NumberOfGAVs2, NumbCr, NumbHr, NumbOr, NumbNr, CpINFlinearr, Linearity).doc_format()
                        ReRun   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent,
                                        Radical, RadfGAVs,
                                        RadQuantity,
                                        Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                        RotoryApp,
                                        SymNumApp,
                                        d2, NumberOfGAVs2, NumbCr, NumbHr, NumbOr, NumbNr, CpINFlinearr, Linearity).rerun_format()
                        Thermoc = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent,
                                          Radical, RadfGAVs,
                                          RadQuantity,
                                          Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                          RotoryApp,
                                          SymNumApp,
                                          d2, NumberOfGAVs2, NumbCr, NumbHr, NumbOr, NumbNr, CpINFlinearr, Linearity).therm_format()
                        Datac   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, "yes",
                                          RadfGAVs, RadQuantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                          RotoryApp, SymNumApp, d2, NumberOfGAVs2, NumbCr, NumbHr, NumbOr, NumbNr, CpINFlinearr, Linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP)
	    
	    
	    
	    
                        # Saving used GAVs in specie and radicals
                        Corr_TestR = list(set(Corr_TestR))
                        for h in range(len(Corr_TestR)):
                            y.write(
                                str(Corr_TestR[h][0]) + "\t" + str(Corr_TestR[h][1]) + "\t" + str(Corr_TestR[h][2]) +
                                "\t" + str(Corr_TestR[h][3]) + "\t" + str(Corr_TestR[h][4]) + "\t" + str(
                                    Corr_TestR[h][5]) +
                                "\t" + str(Corr_TestR[h][6]) + "\t" + str(Corr_TestR[h][7]) + "\t" + str(
                                    Corr_TestR[h][8]) +
                                "\t" + str(Corr_TestR[h][9]) + "\n")
                        # beginning of saving missed GAVs
                        if len(NewCorr_Test) == 0 and len(NewCorr_Test2) > 0:
                            y.write("# missing GAVs:\n")
                            for j in range(len(NewCorr_Test2)):
                                y.write("# " + str(NewCorr_Test2[j]) + "\n")
                        elif len(NewCorr_Test) > 0 and len(NewCorr_Test2) > 0:
                            Dockers = []
                            for k in range(len(NewCorr_Test)):
                                NewCorr_Test2.remove(NewCorr_Test[k])
                            for q in range(len(NewCorr_Test2)):
                                y.write("# " + str(NewCorr_Test2[q]) + "\n")
	    
                        else:
                            pass
                        print("\t Please find your data in '\Therm23\OutputsDir\' directory\n")
                        # end of saving missed GAVs
                    elif str(SaveData).lower() in ["no", "n"]:
                        print("\t Data was not saved.")
                    else:
                        print(
                            "\t Not a proper answer, not saving data and moving to next specie/radical\n\t to exit program type 'end'.")
                    RadfGAVs.clear()
                    RadQuantity.clear()
                    #AllGAVSInter.clear()
                    del RadNumberOfGAVs
            elif RadicalQ.lower() in ["n", "no"]:
                pass
            else:
                print("\t Not a proper answer, skipping radicals moving to")
            print("\t",".." * 46)
            # ******************************************************************************
            # 
            # **********************************************
            print("\t Next species...\n\n")
            # get_ipython().magic('reset -sf')
    except:
	        # =============================================================
            z.write("END")
            z.close()
            y.close()
            f.close()
            t.close()
            q.close()
    #
    print("\t",".." * 46)
    print("\t Note: Please find your data in \Therm23\OutputsDir directory")
    print("\t",".." * 46) 
# *************************************************************
# Switching to Automatic mode
# =============================================================
elif MainMenu == 2:
    #
    print("\t Calculating new thermochemistry in automatic mode")
    #
    Filename = input("\t Please provide the output file name:\n\t ") or str(d1)
    Filename = str(Filename).strip()
    Variable2Test(Filename)
    ## 
    WordFile  = Filename + ".doc"
    DatFile   = Filename + ".dat"
    LSTFile   = Filename + ".LST"
    TmpFile   = Filename + ".tmp"
    RerunFile = Filename + ".rerun"
    # 
    dfObj = Parser(GroupFiles).data()
    # *******************************
    # Switching to Interactive mode
    # *******************************    
    DocFileOut   = OUTPUT + psep + WordFile
    DatFileOut   = OUTPUT + psep + DatFile
    LSTFileOut   = OUTPUT + psep + LSTFile
    TmpFileOut   = OUTPUT + psep + TmpFile
    RerunFileOut = OUTPUT + psep + RerunFile
    # *************************************************************
    # Switching to Interactive mode
    # =============================================================

    f = open(DocFileOut, "w")
    z = open(DatFileOut, "w")
    z.write("THERMO\n")
    z.write("   300.00  1000.00  5000.00\n")
    q = open(LSTFileOut, "w")
    q.write("Units:\tK,calories\n")
    q.write("SPECIES                   \tHf        \tS     \tCP 300    \t400       \t500       \t600       \t800       \t1000      \t1500      \tDATE    \tELEMENTS  \tC   \tH   \tO   \tN   \tRotor \n")
    y = open(TmpFileOut, "w")
    y.write("DATE:\t" + str(d1) + "\n")
    y.write("GROUPNAME                 Hf     S     CP   300     400     500     600     800     1000     1500\n")
    t = open(RerunFileOut, "w")
    t.write("SpecieName          \tFormula          \tNumberOfGroups   \tGroupID   \tQuantity   \tSymmetryNumber   \tNumberOfRotors   \tIfRad   \tParentMol   \tC   \tH   \tO   \tN   \tLinearity\n")
    #
    InChI_database = pd.read_csv(os.path.join(dir_path, relativeGroup, "Database.dat"), sep="\t", engine="python", comment="!")

    print("\t","--" * 46)
    print("\t List of input files available to run in automatic mode:")
    print("\t","==" * 46)
    ListOfReRuns = []
    for u in range(len(InputFile)):
        print("\t", str(InputFile[u]).split(psep)[-1])
    print("\t","==" * 46)
    print("\t","==" * 46)
    print("\t How many input files would you like to use? (Provide an integer or if all use the keyword 'all')")
    print("\t If you don't provide a name and only press 'Enter' the code will run the first file in the list")
    print("\t",".." * 46)
    #
    while True:
        HowMany2Plot = input("\t ") or 0
        Names2Plot   = []
        try:
            HowMany2Plot     = int(HowMany2Plot)
            if  HowMany2Plot == 0:
                Names2Plot.append(InputFile[0])
                break
            else:
                for m in range(HowMany2Plot):
                    print("\t Type #" + str(m+1) + " input file's name to use:   ")
                    FilenameX  = input("\t ")
                    FilenameX  = FilenameX.strip()
                    IndexInput = InputFile.index(CWDPath + psep + FilenameX)
                    Names2Plot.append(InputFile[IndexInput])
                break

        except:
            if   HowMany2Plot.lower() == "all":
                for m in range(len(InputFile)):
                    Names2Plot.append(InputFile[m])
                break
            elif HowMany2Plot.lower() in ["quit", "q", "end"]:
                sys.exit(1)
            else:
                print("\t Wrong answer, try again... ")
                print("\t Only 'all' keyword is allow as input here. ")
                print("\t type 'quit' or 'end' to finish the program.")
                continue

    #
    ########
    InputFile = Names2Plot
    for x in InputFile:
        print("\t","==" * 46)
        print("\t |> Running in Automatic mode from input file:")
        print("\t","==" * 46)
        print("\t |> " + x)
        InputFile = pd.read_csv(x, sep="\t", engine="python", comment="!")
        NewCorr_H = []
        NewCorr_S = []
        NewCorr_Test = []
        AllGAVsInput = []
        for k in range(len(InputFile)):
            print("\t","__" * 46)
            #
            BPRow  = (InputFile.iloc[k,:]).tolist()
            if   x.endswith(".InChI"):
                 SMILES = (BPRow[0].strip()).upper() 
                 INCHI  = (BPRow[1].strip()) 
                 print(SMILES)
                 print(INCHI)
                 for  t in range(len(InChI_database)):
                    if SMILE == InChI_database.iloc[k, 1] or INCHI == InChI_database.iloc[k, 0]:
                       Speciename = InChI_database.iloc[k, 2]
                 else:
                    pass
                 Formula      = BPRow[1] 
                 Formula      = Formula.strip() 
                 Formula      = Formula.upper()
                 if Formula.upper() == "NAN":
                    Formula   = InChI_inputFile.iloc[k, 4]
                 else:
                    pass
                 NumberOfGAVs = int(BPRow[2]) 
                 if str(NumberOfGAVs).upper() == "NAN":
                    NumberOfGAVs   = int(InChI_inputFile.iloc[k, 5])
                 else:
                    pass
                 GroupsApp    = BPRow[3] 
                 GroupsApp    = GroupsApp.strip() 
                 if GroupsApp.upper() == "NAN":
                    GroupsApp   = InChI_inputFile.iloc[k, 6]
                 else:
                    pass
                 QuantyApp    = BPRow[4]  
                 QuantyApp    = str(QuantyApp).strip()  
                 if str(QuantyApp).upper() == "NAN":
                    QuantyApp   = InChI_inputFile.iloc[k, 7]
                 else:
                    pass
                 SymNumApp    = int(BPRow[5]) 
                 if str(SymNumApp).upper() == "NAN":
                    SymNumApp   = int(InChI_inputFile.iloc[k, 8])
                    print(SymNumApp)
                 else:
                    pass
                 RotoryApp    = int(BPRow[6]) 
                 if str(RotoryApp).upper() == "NAN":
                    RotoryApp   = int(InChI_inputFile.iloc[k, 9])
                 else:
                    pass
                 BooleanRad   = BPRow[7].strip()       
                 BooleanRad   = BooleanRad.upper()       
                 if str(BooleanRad).upper() == "NAN":
                    BooleanRad   = InChI_inputFile["IfRad"]
                 else:
                    pass
                 pparent      = BPRow[8].strip()
                 Parent       = pparent.upper()
                 if str(Parent).upper() == "NAN":
                    Parent   = InChI_inputFile["ParentMol"]
                 else:
                    pass
                 NumbC        = int(BPRow[9])
                 if str(NumbC).upper() == "NAN":
                    NumbC   = int(InChI_inputFile["C"])
                 else:
                    pass
                 NumbH        = int(BPRow[10])
                 if str(NumbH).upper() == "NAN":
                    NumbH   = int(InChI_inputFile["H"])
                 else:
                    pass
                 NumbO        = int(BPRow[11])
                 if str(NumbO).upper() == "NAN":
                    NumbO   = int(InChI_inputFile["O"])
                 else:
                    pass
                 NumbN        = int(BPRow[12])
                 if str(NumbN).upper() == "NAN":
                    NumbN   = int(InChI_inputFile["N"])
                 else:
                    pass
                 Linearity    = BPRow[13].strip()   
                 Linearity    = Linearity.upper() 
                 if str(Linearity).upper() == "NAN":
                    Linearity   = InChI_inputFile["Linearity"]
                 else:
                    pass            
            elif x.endswith(".inp"):
                 Speciename   = BPRow[0] 
                 Speciename   = Speciename.strip() 
                 Formula      = BPRow[1] 
                 Formula      = Formula.strip() 
                 Formula      = Formula.upper() 
                 NumberOfGAVs = int(BPRow[2]) 
                 GroupsApp    = BPRow[3] 
                 GroupsApp    = GroupsApp.strip() 
                 QuantyApp    = BPRow[4]  
                 QuantyApp    = str(QuantyApp).strip()  
                 SymNumApp    = int(BPRow[5]) 
                 RotoryApp    = int(BPRow[6]) 
                 BooleanRad   = BPRow[7].strip()       
                 BooleanRad   = BooleanRad.upper()       
                 pparent      = BPRow[8].strip()
                 Parent       = pparent.upper()
                 NumbC        = int(BPRow[9])
                 NumbH        = int(BPRow[10])
                 NumbO        = int(BPRow[11])
                 NumbN        = int(BPRow[12])
                 Linearity    = BPRow[13].strip()   
                 Linearity    = Linearity.upper()   
            else:                
                 print("\t Please provide either an '.inp' or 'InChI' valid file.")                
                 sys.exit(1)                

            #
            print("\t Summarising")
            print("\t SpecieName: {0}".format(Speciename))
            print("\t Formula: {0}".format(Formula))
            print("\t Number of GAVs: {0}".format(NumberOfGAVs))
            print("\t #GAV - GAV and Quantity: ")
            GAVs = str(GroupsApp).split(",")
            Quantity = str(QuantyApp).split(",")
            for j in range(len(GAVs)):
                if   len(GAVs[j]) == 1:
                   print("\t "+str(j+1) +"    - {0}                    - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 2:
                   print("\t "+str(j+1) +"    - {0}                   - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 3:
                   print("\t "+str(j+1) +"    - {0}                  - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 4:
                   print("\t "+str(j+1) +"    - {0}                 - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 5:
                   print("\t "+str(j+1) +"    - {0}                - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 6:
                   print("\t "+str(j+1) +"    - {0}               - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 7:
                   print("\t "+str(j+1) +"    - {0}              - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 8:
                   print("\t "+str(j+1) +"    - {0}             - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 9:
                   print("\t "+str(j+1) +"    - {0}            - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 10:
                   print("\t "+str(j+1) +"    - {0}           - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 11:
                   print("\t "+str(j+1) +"    - {0}          - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 12:
                   print("\t "+str(j+1) +"    - {0}         - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 13:
                   print("\t "+str(j+1) +"    - {0}        - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 14:
                   print("\t "+str(j+1) +"    - {0}       - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 15:
                   print("\t "+str(j+1) +"    - {0}      - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 16:
                   print("\t "+str(j+1) +"    - {0}     - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 17:
                   print("\t "+str(j+1) +"    - {0}    - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 18:
                   print("\t "+str(j+1) +"    - {0}   - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 19:
                   print("\t "+str(j+1) +"    - {0}  - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 20:
                   print("\t "+str(j+1) +"    - {0} - {1}".format(GAVs[j], Quantity[j]))
                elif len(GAVs[j]) == 0:
                   print("\t WARNING! Error! Something went erong! Not GAV found!")
                   sys.exit(1)
                else:
                   print("\t "+str(j+1) +"    - {0} -\t{1}".format(GAVs[j], Quantity[j]))
                AllGAVsInput.append(GAVs[j])
            print("\t Number of rotors: {0}".format(RotoryApp))
            print("\t Symmetry number: {0}".format(SymNumApp))
            print("\t","__" * 46)
            # ..................
            if  Linearity.lower() in ["no", "n"]:
                Cp0linear   = round((3.5) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
            elif Linearity.lower() in ["yes", "y"]:
                Cp0linear   = round((4.0) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 2.0) * (1.9), 2)

            # ..................
            if BooleanRad.lower() in ["no", "n"]:
                (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                         speciesname=Speciename,
                                                                                                         formula=Formula,
                                                                                                         numberofgroups=NumberOfGAVs,
                                                                                                         groupid=GAVs,
                                                                                                         quantity=Quantity,
                                                                                                         symmetrynumber=SymNumApp,
                                                                                                         numberofrotors=RotoryApp
                                                                                                         ).thermo_props()
            elif BooleanRad.lower() in ["yes", "y"]:
                (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                         speciesname=Speciename,
                                                                                                         formula=Formula,
                                                                                                         numberofgroups=NumberOfGAVs,
                                                                                                         groupid=GAVs,
                                                                                                         quantity=Quantity,
                                                                                                         symmetrynumber=SymNumApp,
                                                                                                         numberofrotors=RotoryApp
                                                                                                         ).thermo_props_rads()
            print("\t Calculations:\n\t")
            print("\t Speciename        \tH(298K)\tS(298K)\tCp300K\t400K\t500K\t600K\t800K\t1000K\t1500K")
            

            cp_str = ("\t" + str(Cp300) + "\t" + str(Cp400) + "\t" + str(Cp500) + "\t" + str(Cp600) 
                      + "\t" + str(Cp800) + "\t" + str(Cp1000) + "\t" + str(Cp1500))
            name_len18 = f"{Speciename[:18]:18}"

            print("\t " + name_len18 + "\t" + str(Enthalpy) + "\t" + str(Entropy) + cp_str)


            print("\t",".." * 46)
            # =========================================================================
            # This block is used to calculate the NASA polynomials
            # =========================================================================
            # *******************************************************************
            # 2nd set of coefficients ; with breaking point at T = 1000 K
            # *******************************************************************
            # function to solve:
	    
            def funcCP(T, a1, a2, a3, a4, a5):
                # Cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
                # CP/R here produce by code
                return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)*R
	    
            def funcH(T, a1, a2, a3, a4, a5, a6):
                T2 = T * T
                T4 = T2 * T2
                # H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
                # H here produce by code
                return (a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T2 * T / 4 + a5 * T4 / 5 + a6 / T) * R * T / 1000 # in kcal units
	    
            def funcS(T, a1, a2, a3, a4, a5, a7):
                import math
                T2 = T * T
                T4 = T2 * T2
                # S/R  = a1 lnT + a2 T + a3 T^2 /2 + a4 T^3 /3 + a5 T^4 /4 + a7
                # S here produce by code
                return (a1 * math.log(T) + a2 * T + a3 * T2 / 2 + a4 * T2 * T / 3 + a5 * T4 / 4 + a7) * R
	    
            Temps = [300, 400, 500, 600, 800, 1000, 1500]

            (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300, cp400=Cp400, cp500=Cp500, cp600=Cp600, cp800=Cp800, cp1000=Cp1000, cp1500=Cp1500, temps=Temps).fit_termo()
            print("\t",".." * 46)

            print_elem_info(NumbC, NumbH, NumbO, NumbN)

            NumberAtoms = round((NumbC + NumbH + NumbO + NumbN), 2)
            print("\t Cp0    = ", Cp0linear)
            print("\t CpINF  = ", CpINFlinear)
            print("\t",".." * 46)
	    	# 
            (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, BooleanRad, GAVs, Quantity, Formula, 
                                                       Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500, RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC,
                                                       NumbH, NumbO, NumbN, CpINFlinear, Linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)		


            # ================================================================================================================================================================
            # Saving data
            # ================================================================================================================================================================
            if str(Formula) in str(InputFile.iloc[k, 1]):
                Doc     = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, BooleanRad,
                                  GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                  RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).doc_format()
                ReRun   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, BooleanRad,
                                  GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                  RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).rerun_format()
                Thermoc = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, BooleanRad,
                                  GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                  RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).therm_format()
                Datac   = Outputs(DocFileOut, DatFileOut, LSTFileOut, RerunFileOut, Speciename, Parent, BooleanRad,
                                  GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                  RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP)
	    
            else:
                pass
            # *****************************************************************************
            Corr_Test = list(ZIPTest)
            for h in range(len(Corr_Test)):
                NewCorr_Test.append(Corr_Test[h])


        CheckingTest = list(set(NewCorr_Test))
        for h in range(len(CheckingTest)):
            y.write(str(CheckingTest[h][0]) + "\t" + str(CheckingTest[h][1]) + "\t" + str(CheckingTest[h][2]) + "\t" + str(CheckingTest[h][3]) + "\t" + str(CheckingTest[h][4]) + "\t" + str(CheckingTest[h][5]) +
                    "\t" + str(CheckingTest[h][6]) + "\t" + str(CheckingTest[h][7]) + "\t" + str(CheckingTest[h][8]) + "\t" + str(CheckingTest[h][9]) + "\n")

        FilterAllGAVsInput = list(set(AllGAVsInput))

        ListaChecker2 = []
        NotFoundGAVS  = []
        for u in range(len(CheckingTest)):
            ListaChecker2.append(CheckingTest[u][0])
        for p in range(len(FilterAllGAVsInput)):
            if FilterAllGAVsInput[p] in ListaChecker2:
                pass
            else:
                NotFoundGAVS.append(FilterAllGAVsInput[p])
        if len(NotFoundGAVS) > 0:
            print(".." * 50)
            y.write("# missing GAVs:\n")
            print("\t Group:  ")
            for w in range(len(NotFoundGAVS)):
                print("\t '" + str(NotFoundGAVS[w]) + "' was not found in GAVs files!")
                y.write("# " + str(NotFoundGAVS[w]) + "\n")
            print("\n\t Please be sure that all GAVs you asked for are included in the files!")
            print("\t Add new file(s) or edit existing ones to add the missing GAV(s) in 'GroupsDir' folder!.")
            print("\t Remember, files must be '*.grp' extension.")
            print("\t Example of file format:\t(don't forget the ',' after the GAV's name)")
            print("\t ")
            print("\t 84       Hf     S   Cp: 300    400    500    600    800   1000   1500")
            print("\t C/C/H3, -10.01  30.29   6.22   7.74   9.24  10.62  12.84  14.59  17.35  SMB optimisation")
            print("\t C/CB/H3,       -10.01  30.29   6.22   7.74   9.24  10.62  12.84  14.59  17.35")
            print("\t we have printed the missing group(s) at '*.tmp' file preceded by a '#' symbol.")
        elif len(NotFoundGAVS) == 0:
            pass
        # ========================================================
    # ========================================================
    # Here a tmp file has been generated to storage all GAVs
    # used during the calculations of this thermochemistry
    # ========================================================
    z.write("END")
    z.close()
    y.close()
    f.close()
    t.close()
    q.close()


    print("\t",".." * 46)
    print("\t Note: Please find your data in \Therm23\OutputsDir directory")
    print("\t",".." * 46)
# *************************************************************
# Switching to Re-calculation mode
# =============================================================
elif MainMenu == 3:
    CWD = os.getcwd()
    print("\t Re-calculating thermochemistry")
    print("\t","--" * 46)
    print("\t List of files available to re-calculate thermochemistry:")
    print("\t","==" * 46)
    ListOfReRuns = []
    for u in range(len(InputFileRe)):
        print("\t", str(InputFileRe[u]).split(psep)[-1])
    print("\t","==" * 46)
    ListOfReRunsDOC = []
    for u in range(len(DocRerun)):
        print("\t", str(DocRerun[u]).split(psep)[-1])
    print("\t","==" * 46)
    print("\t Choose one and insert its name as requested. However, if your file is not listed")
    print("\t please copy and paste it into '/Therm23/OutputsDir/' directory")
    print("\t file should have an extension '.rerun' or '.doc', i.e. Test_25_12_2021.rerun")
    print("\t",".." * 46)
    #
    CPath = os.getcwd()
    OUTPUT = CPath + "/OutputsDir"
    Filename = input("\t Please provide the file name:\t ")
    Variable2Test(Filename)
    name, extension = os.path.splitext(Filename)

    if Filename.endswith(".rerun"):
       if (len(Filename)) == 0:
           print("\t Not file's name was given, closing program.")
           sys.exit(1)
       else:
           Filename = Filename.strip()
           Filename = str(Filename).split(".rerun")[0]
           if Filename + ".tmp" in TMPinputs2:

               TMPfile1 = pd.read_csv(CWD + "/OutputsDir/" + Filename + ".tmp", sep="\t", skiprows=[0], engine="python")
           else:
               print("\t temporal file (*.tmp) couldn't be found... closing program now, try again.")
               sys.exit(1)
       # Insert file to re-run thermochemistry calculations based on GAVs
       print("\t Reading...  %s" % Filename + ".rerun")
       # **************************************************
       # Beginning of thermochemistry re-calculations loop
       # **************************************************
       dfObj = Parser(GroupFiles).data()
       # *******************************
       # Reading content and printing on screen
       PathOuputFiles = CWD + "/OutputsDir/" + Filename
       # Switching to Interactive mode
       # *******************************
       NamesFilesInp    = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".rerun"
       NamesFilesDoc    = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".doc"
       NamesFilesTherm  = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".LST"
       ThermoFiles      = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".dat"
       TemporalDataBase = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".tmp"
       # *************************************************************
       # Switching to Interactive mode
       # =============================================================

       ########
       f = open(NamesFilesDoc, "w")
       t = open(NamesFilesInp, "w")
       q = open(NamesFilesTherm, "w")
       z = open(ThermoFiles, "w")
       y = open(TemporalDataBase, "w")
       q.write("Units:\tK,calories\n")
       z.write("THERMO\n")
       z.write("\t300.00\t1000.00\t5000.00\n")
       t.write("SpecieName\tFormula\tNumberOfGroups\tGroupID\tQuantity\tSymmetryNumber\tNumberOfRotors\tIfRad\tParentMol\tC\tH\tO\tN\tLinearity\n")
       q.write("SPECIES\t\t\t Hf  \tS\tCP  300\t\t400\t\t500\t\t600\t\t800\t\t1000\t1500\tDATE     \tELEMENTS\n")
       y.write("DATE:\t" + str(d1) + "\n")
       y.write("SPECIES\tHf\tS\tCP300\t400\t500\t600\t800\t1000\t1500\n")
       ########
       print("\t","==" * 46)
       print("\t> Running in Automatic mode.")
       print("\t","==" * 46)

       InputFile    = pd.read_csv(PathOuputFiles + ".rerun", sep="\t", engine="python", comment="!")
       NewCorr_H    = []
       NewCorr_S    = []
       NewCorr_Test = []
       AllGAVsInput = []
       # ---------------
       # A nested loop:
       # ---------------
       for k in range(len(InputFile)):
           print("\t","__" * 46)
           Speciename   = InputFile.iloc[k, 0]
           Formula      = InputFile.iloc[k, 1]
           Formula      = str(Formula).upper()
           NumberOfGAVs = InputFile.iloc[k, 2]
           GroupsApp    = InputFile.iloc[k, 3]
           QuantyApp    = InputFile.iloc[k, 4]
           SymNumApp    = InputFile.iloc[k, 5]
           RotoryApp    = InputFile.iloc[k, 6]
           BooleanRad   = str(InputFile.iloc[k, 7]).upper()
           pparent      = InputFile.iloc[k, 8]
           NumbC        = InputFile.iloc[k, 9]
           NumbH        = InputFile.iloc[k, 10]
           NumbO        = InputFile.iloc[k, 11]
           NumbN        = InputFile.iloc[k, 12]
           Parent       = pparent.upper()
           Linearity    = InputFile.iloc[k, 13]
           Linearity    = Linearity.upper()
           #################
           #print(Formula)
           #print(NumberOfGAVs)
           #print(GroupsApp)
           #print(QuantyApp)
           #################
           print("\t SpecieName: {0}".format(Speciename))
           print("\t Formula: {0}".format(Formula))
           print("\t Number of GAVs: {0}".format(NumberOfGAVs))
           print("\t GAVs and Quantity: ")
           GAVs = str(GroupsApp).split(",")
           Quantity = str(QuantyApp).split(",")
           for j in range(len(GAVs)):
               print("\t {0}\t{1}".format(GAVs[j], Quantity[j]))
               AllGAVsInput.append(GAVs[j])
           print("\t Number of rotors: {0}".format(RotoryApp))
           print("\t Symmetry number: {0}".format(SymNumApp))
           print("\t","__" * 46)
           # ..................
           if   Linearity.lower() in ["no", "n"]:
                Cp0linear = round((3.5) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
           elif Linearity.lower() in ["yes", "y"]:
                Cp0linear = round((4.0) * (1.9), 2)
                CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 2.0) * (1.9), 2)
           # ..............................................................................


           # ..................
           if BooleanRad.lower() in ["no", "n"]:
               (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                        speciesname=Speciename,
                                                                                                        formula=Formula,
                                                                                                        numberofgroups=NumberOfGAVs,
                                                                                                        groupid=GAVs,
                                                                                                        quantity=Quantity,
                                                                                                        symmetrynumber=SymNumApp,
                                                                                                        numberofrotors=RotoryApp
                                                                                                        ).thermo_props()
           elif BooleanRad.lower() in ["yes", "y"]:
               (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                        speciesname=Speciename,
                                                                                                        formula=Formula,
                                                                                                        numberofgroups=NumberOfGAVs,
                                                                                                        groupid=GAVs,
                                                                                                        quantity=Quantity,
                                                                                                        symmetrynumber=SymNumApp,
                                                                                                        numberofrotors=RotoryApp
                                                                                                        ).thermo_props_rads()
           print("\t Calculations:\n\t")
           print("\t Formula H(298K) S(298K) Cp300K Cp400K Cp500K Cp600K Cp800K Cp1000K Cp1500K date")
           print(("\t " + str(Formula) + " " + str(round(Enthalpy,2)) + " " + str(round(Entropy,2)) + " " + str(round(Cp300,2)) + " " + 
                str(round(Cp400,2)) + " " + str(round(Cp500,2)) + " " + str(round(Cp600,2)) + " " + str(round(Cp800,2)) + " " + 
                str(round(Cp1000,2)) + " " + str(round(Cp1500,2)) + " " + str(d2) + "\n"))
           print("\t",".." * 46)
		   
           # =========================================================================
           # This block is used to calculate the NASA polynomials
           # =========================================================================
		   
           # *******************************************************************
           # 2nd set of coefficients ; with breaking point at T = 1000 K
           # *******************************************************************
           # function to solve:
		   
           def funcCP(T, a1, a2, a3, a4, a5):
               # Cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
               # CP/R here produce by code
               return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3 + a5 * (T) ** 4)
		   
           def funcH(T, a1, a2, a3, a4, a5, a6):
               T2 = T * T
               T4 = T2 * T2
               # H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
               # H here produce by code
               return (a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T2 * T / 4 +
                       a5 * T4 / 5 + a6 / T) * R * T / 1000 #in kcal unit
		   
           def funcS(T, a1, a2, a3, a4, a5, a7):
               import math
               T2 = T * T
               T4 = T2 * T2
               # S/R  = a1 lnT + a2 T + a3 T^2 /2 + a4 T^3 /3 + a5 T^4 /4 + a7
               # S here produce by code
               return (a1 * math.log(T) + a2 * T + a3 * T2 / 2 + a4 * T2 * T / 3 + a5 * T4 / 4 + a7) * R
		   
           Temps = [300, 400, 500, 600, 800, 1000, 1500]
           (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                 cp400=Cp400,
                                                 cp500=Cp500,
                                                 cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                 cp1500=Cp1500, temps=Temps).fit_termo()
           print("\t",".." * 46)

           print_elem_info(NumbC, NumbH, NumbO, NumbN)

           NumberAtoms = round((NumbC + NumbH + NumbO + NumbN), 2)
           print("\t Cp0    = ", Cp0linear)
           print("\t CpINF  = ", CpINFlinear)
           print("\t",".." * 46)
		   # 
           (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                             GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                             RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)		
           # ================================================================================================================================================================
           # Saving data
           # ================================================================================================================================================================
           if str(Formula) in str(InputFile.iloc[k, 1]):
               Doc     = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).doc_format()
               ReRun   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).rerun_format()
               Thermoc = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).therm_format()
               Datac   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, Linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP)


           else:
               pass
           # *****************************************************************************
           # *****************************************************************************
           # Start of DAT file to save NASA polynomials
           # *****************************************************************************           # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           print("\t",".." * 46)
           Corr_Test = list(ZIPTest)
           for h in range(len(Corr_Test)):
               NewCorr_Test.append(Corr_Test[h])
           # ===============================================================
       CheckingTest = list(set(NewCorr_Test))
       for h in range(len(CheckingTest)):
           y.write(str(CheckingTest[h][0]) + "\t" + str(CheckingTest[h][1]) + "\t" + str(CheckingTest[h][2]) +
                   "\t" + str(CheckingTest[h][3]) + "\t" + str(CheckingTest[h][4]) + "\t" + str(CheckingTest[h][5]) +
                   "\t" + str(CheckingTest[h][6]) + "\t" + str(CheckingTest[h][7]) + "\t" + str(CheckingTest[h][8]) +
                   "\t" + str(CheckingTest[h][9]) + "\n")
           # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       # *******************************************************************
       # ========================================================
       z.write("END")
       z.close()
       y.close()
       f.close()
       t.close()
       q.close()
       CheckingTest = list(set(NewCorr_Test))

       #
       #
       # Searching for GAVs names one by one, if it finds a coincidence it compares values if not skip it.
       def GAVsChecker(a, b, c, message):
           if float(CheckingTest[a][b]) == float(TMPfile1.iloc[c, b]):
               pass
           else:
               NotDuplicates.append((CheckingTest[a][0], message, CheckingTest[a][b], TMPfile1.iloc[c, b]))
       #
       NotDuplicates = []
       for f in range(len(CheckingTest)):
           for k in range(len(TMPfile1)):
               if CheckingTest[f][0] == TMPfile1.iloc[k, 0]:
                   GAVsChecker(f, 1, k, "H")
                   GAVsChecker(f, 2, k, "S")
                   GAVsChecker(f, 3, k, "300")
                   GAVsChecker(f, 4, k, "400")
                   GAVsChecker(f, 5, k, "500")
                   GAVsChecker(f, 6, k, "600")
                   GAVsChecker(f, 7, k, "800")
                   GAVsChecker(f, 8, k, "1000")
                   GAVsChecker(f, 9, k, "1500")
               else:
                   pass
       #
       docfile1 = CWD + "/OutputsDir/" + Filename + ".doc"
       rerunfile1 = CWD + "/OutputsDir/" + Filename + ".rerun"
       thermofile1 = CWD + "/OutputsDir/" + Filename + ".LST"
       tmpfile1 = CWD + "/OutputsDir/" + Filename + ".tmp"
       datfile1 = CWD + "/OutputsDir/" + Filename + ".dat"
       # Cheking if we got any duplicated GAVs or not
       if len(NotDuplicates) == 0:
           pass

       else:
           print("\t There is/are updated GAVs available")
           print("\t GAV's name\t\tProperty\tNew value\tOld value")
           for h in range(len(NotDuplicates)):
               print("\t " + str(NotDuplicates[h][0]) + "  \t\t" + str(NotDuplicates[h][1]) + "\t\t\t" + 
                     str(NotDuplicates[h][2]) + "\t\t" + str(NotDuplicates[h][3]))
           print("\t This code automatically will calculate using the new values and keep you old files too")
           print(
               "\t Please provide with a name for these output files or just hit Enter to let the code to randomly name it")
           Random1st = (random.randrange(1, 100))
           Random2nd = (random.randrange(1, 100))
           Random3rd = (random.randrange(1, 100))
           NewName = input("\t Please type a file name: ") or (str(d1) + "_" + str(Random1st) + str(Random2nd) + str(Random3rd))
           FilrOutpuNewName = CWD + "/OutputsDir/" + NewName
           print("\t New files named as follows (i.e. Date_RandomNumber; 30_07_2021_12345): \n\t " +
                 str(NewName).split(psep)[-1])
           os.rename(NamesFilesDoc, FilrOutpuNewName + ".doc")
           os.rename(NamesFilesInp, FilrOutpuNewName + ".rerun")
           os.rename(NamesFilesTherm, FilrOutpuNewName + ".LST")
           os.rename(TemporalDataBase, FilrOutpuNewName + ".tmp")
           os.rename(ThermoFiles, FilrOutpuNewName + ".dat")
    elif Filename.endswith(".DOC") or Filename.endswith(".doc") or Filename.endswith(".DOCX")or Filename.endswith(".docx"):
       print("\t This is a doc file")
       print("\t Reading...  %s" % Filename)
       # **************************************************
       # Beginning of thermochemistry re-calculations loop
       # **************************************************
       dfObj = Parser(GroupFiles).data()
       # *******************************
       # Reading content and printing on screen
       PathOuputFiles = CWD + "/OutputsDir/" + Filename
       # Switching to Interactive mode
       # *******************************
       NamesFilesInp    = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".rerun"
       NamesFilesDoc    = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".doc"
       NamesFilesTherm  = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".LST"
       ThermoFiles      = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".dat"
       TemporalDataBase = OUTPUT + psep + str(Filename) + "R_" + str(d1) + ".tmp"
       # *************************************************************
       # Switching to Interactive mode
       # =============================================================
       ########
       f = open(NamesFilesDoc, "w")
       t = open(NamesFilesInp, "w")
       q = open(NamesFilesTherm, "w")
       z = open(ThermoFiles, "w")
       y = open(TemporalDataBase, "w")
       q.write("Units:\tK,calories\n")
       z.write("THERMO\n")
       z.write("\t300.00\t1000.00\t5000.00\n")
       t.write("SpecieName          \tFormula          \tNumberOfGroups   \tGroupID   \tQuantity   \tSymmetryNumber   \tNumberOfRotors   \tIfRad   \tParentMol   \tC   \tH   \tO   \tN   \tLinearity\n")
       q.write("SPECIES\t\t\t Hf  \tS\tCP  300\t\t400\t\t500\t\t600\t\t800\t\t1000\t1500\tDATE     \tELEMENTS\n")
       y.write("DATE:\t" + str(d1) + "\n")
       y.write("SPECIES\tHf\tS\tCP300\t400\t500\t600\t800\t1000\t1500\n")
       ########
       print("\t","==" * 46)
       print("\t Reading from file:")
       print("\t " + PathOuputFiles)
       print("\t","==" * 46)
       InputFile = pd.read_csv(PathOuputFiles, names=['col'], sep="s%", engine="python", comment="#")

       NewCorr_H = []
       NewCorr_S = []
       NewCorr_Test = []

       index = InputFile.index
       try:
           condition    = InputFile["col"] == "SPECIES"
           condition2   = InputFile["col"] == "ENDSPECIES"
       except:          
           condition    = InputFile["col"] == "SpecieName"
           condition2   = InputFile["col"] == "Endspecies"
       species_indices  = index[condition]
       species_indices2 = index[condition2]
       species_indices_list  = species_indices.tolist()
       species_indices_list2 = species_indices2.tolist()

       Question1 = int(input("\t Type 1: molecules are considered non linear and group values are used without change\n\t Type 2: user provides linearity and any extra groups before calculation\n\t "))
       if Question1 == 1:
          for k in species_indices_list: 
              print("\t Index line #:\t " + str(k))

              u = species_indices_list.index(k)
              CheckIn = str(InputFile.iloc[k+2]).split()

              GroupsApp = []
              QuantyApp = []

              if  "radical" in CheckIn:

                  SPECIENAME   = ((InputFile.iloc[k+1])[-1])
                  FORMULA      = ((InputFile.iloc[k+3]).str.split())[-1][-1]
                  PARENT       = ((InputFile.iloc[k+5]).str.split())[-1][-1]
                  try:
                      LIN          = ((InputFile.iloc[k+7]).str.split())[-1][-2]
                  except:
                      pass
                  NumberOfGAVs = ((InputFile.iloc[k+8]).str.split())[-1][-1]
                  #HeadGAVs     = ((InputFile.iloc[k+10]))
                  BooleanRad   = "YES"
                  GAVsCounter  = 0
                  while GAVsCounter < int(NumberOfGAVs):
                        HeadGAVs    = ((InputFile.iloc[k+10]))
                        if "|" in (HeadGAVs)[-1]:
               
                           BarSplitted   = (HeadGAVs)[-1].split("|")

                           NumbGAV1  = ((BarSplitted[0]).split("-")[0]).strip()
                           GAVName1  = ((BarSplitted[0]).split("-")[1]).strip()
                           GAVtimes1 = ((BarSplitted[0]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName1)
                           QuantyApp.append(GAVtimes1)
                           
                           NumbGAV2  = ((BarSplitted[1]).split("-")[0]).strip()
                           GAVName2  = ((BarSplitted[1]).split("-")[1]).strip()
                           GAVtimes2 = ((BarSplitted[1]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName2)
                           QuantyApp.append(GAVtimes2)
                           
                           k += 1
                           GAVsCounter += 2
                        else:
                            
                           NumbGAV  = (HeadGAVs)[-1].split("-")[0]
                           GAVName  = (HeadGAVs)[-1].split("-")[1]
                           GAVtimes = (HeadGAVs)[-1].split("-")[-1]
                           GroupsApp.append(GAVName)
                           QuantyApp.append(GAVtimes)
                           
                           k += 1
                           GAVsCounter += 1                  
                  Parent = PARENT.upper()
                  RotoryApp  = InputFile.iloc[species_indices_list2[u]-6]
                  SymNumApp  = InputFile.iloc[species_indices_list2[u]-5]
                  
              # ***********************************************************
              elif  "diradical" in CheckIn:
                
                  SPECIENAME   = ((InputFile.iloc[k+1])[-1])
                  FORMULA      = ((InputFile.iloc[k+3]).str.split())[-1][-1]
                  PARENT       = ((InputFile.iloc[k+5]).str.split())[-1][-1]
                  try:
                      LIN          = ((InputFile.iloc[k+7]).str.split())[-1][-2]
                  except:
                      pass
                  NumberOfGAVs = ((InputFile.iloc[k+8]).str.split())[-1][-1]
                  
                  BooleanRad   = "YES"
                  GAVsCounter  = 0
                  while GAVsCounter < int(NumberOfGAVs):
                        HeadGAVs    = ((InputFile.iloc[k+10]))
                        if "|" in (HeadGAVs)[-1]:         
                           BarSplitted   = (HeadGAVs)[-1].split("|")
                           NumbGAV1  = ((BarSplitted[0]).split("-")[0]).strip()
                           GAVName1  = ((BarSplitted[0]).split("-")[1]).strip()
                           GAVtimes1 = ((BarSplitted[0]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName1)
                           QuantyApp.append(GAVtimes1)
                           
                           NumbGAV2  = ((BarSplitted[1]).split("-")[0]).strip()
                           GAVName2  = ((BarSplitted[1]).split("-")[1]).strip()
                           GAVtimes2 = ((BarSplitted[1]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName2)
                           QuantyApp.append(GAVtimes2)
                           
                           k += 1
                           GAVsCounter += 2
                        else:
                            
                           NumbGAV  = (HeadGAVs)[-1].split("-")[0]
                           GAVName  = (HeadGAVs)[-1].split("-")[1]
                           GAVtimes = (HeadGAVs)[-1].split("-")[-1]
                           GroupsApp.append(GAVName)
                           QuantyApp.append(GAVtimes)
                           
                           k += 1
                           GAVsCounter += 1                  
                  Parent = PARENT.upper()
                  RotoryApp  = InputFile.iloc[species_indices_list2[u]-7]
                  SymNumApp  = InputFile.iloc[species_indices_list2[u]-6]
                  
              # ***********************************************************
              elif "molecule" in CheckIn:

                  SPECIENAME   = (InputFile.iloc[k+1])[-1]
                  FORMULA      = ((InputFile.iloc[k+3]).str.split())[-1][-1]
                  try:
                      LIN          = ((InputFile.iloc[k+4]).str.split())[-1][-2]
                  except:
                      pass
                  NumberOfGAVs = ((InputFile.iloc[k+5]).str.split())[-1][-1]
                  BooleanRad = "NO"
                  GAVsCounter = 0
                  while GAVsCounter < int(NumberOfGAVs):
                        HeadGAVs    = ((InputFile.iloc[k+7]))
                        if "|" in (HeadGAVs)[-1]:            
                           BarSplitted   = (HeadGAVs)[-1].split("|")
                           NumbGAV1  = ((BarSplitted[0]).split("-")[0]).strip()
                           GAVName1  = ((BarSplitted[0]).split("-")[1]).strip()
                           GAVtimes1 = ((BarSplitted[0]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName1)
                           QuantyApp.append(GAVtimes1)
                           NumbGAV2  = ((BarSplitted[1]).split("-")[0]).strip()
                           GAVName2  = ((BarSplitted[1]).split("-")[1]).strip()
                           GAVtimes2 = ((BarSplitted[1]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName2)
                           QuantyApp.append(GAVtimes2)
                           k += 1
                           GAVsCounter += 2
                        else:
                            
                           NumbGAV  = (HeadGAVs)[-1].split("-")[0]
                           GAVName  = (HeadGAVs)[-1].split("-")[1]
                           GAVtimes = (HeadGAVs)[-1].split("-")[-1]
                           GroupsApp.append(GAVName)
                           QuantyApp.append(GAVtimes)
                           k += 1
                           GAVsCounter += 1
                  PARENT     = "NO" 
                  Parent     = PARENT.upper()
                  RotoryApp  = InputFile.iloc[species_indices_list2[u]-3]
                  SymNumApp  = InputFile.iloc[species_indices_list2[u]-2]
              GroupsApp = [x.strip() for x in GroupsApp]
              GroupsApp = ",".join(GroupsApp)
              QuantyApp = [x.strip() for x in QuantyApp]
              QuantyApp = ",".join(QuantyApp)
              # Block added 
              Speciename   = SPECIENAME
              Formula      = FORMULA
              Formula      = str(Formula).upper()
              Formula      = str(Formula).strip()
              NumberOfGAVs = NumberOfGAVs
              GroupsApp    = GroupsApp
              QuantyApp    = QuantyApp
              SymNumApp2    = (SymNumApp[-1]).split()
              SymNumApp    = int(SymNumApp2[-1])
              RotoryApp    = int(RotoryApp[-1][-1])
              BooleanRad   = BooleanRad.upper()

              try:
                  if   LIN == "NONLINEAR":
                       linearity    = "NO"
                  elif LIN == "LINEAR":
                       linearity    = "YES"
                  else:
                       linearity    = "NO"
              except:
                  linearity = "NO"
              print(linearity)
              ManyC = Formula.count('C')
              ManyH = Formula.count('H')
              ManyO = Formula.count('O')
              ManyN = Formula.count('N')

              SumC = []
              SumH = []
              SumO = []
              SumN = []
              if ManyC == 0:
                 SumC.append(0)
              else:
                 pass
              if ManyH == 0:
                 SumH.append(0)
              else:
                 pass
              if ManyO == 0:
                 SumO.append(0)
              else:
                 pass
              if ManyN == 0:
                 SumN.append(0)
              else:
                 pass
              for l in range(len(Formula)):
                   if str(Formula[l]).isalpha() == True:
                       if   str(Formula[l]) == "C":
                            SumC.append(1)                     
                       elif str(Formula[l]) == "H":
                            SumH.append(1)                       
                       elif str(Formula[l]) == "O":
                            SumO.append(1)                       
                       elif str(Formula[l]) == "N":
                            SumN.append(1)
                       else:
                            pass                    
                   elif str(Formula[l]).isnumeric() == True:
                       NumIndex = l

                       if   Formula[NumIndex-1] == "C":
                              SumC.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-2] == "C":
                              SumC[-1] = int(str(Formula[l-1])+str(Formula[l]))-1 

                       elif Formula[NumIndex-1] == "H":
                              SumH.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-2] == "H":
                              SumH[-1] = int(str(Formula[l-1])+str(Formula[l]))-1 
                       elif Formula[NumIndex-1] == "O":
                              SumO.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-2] == "O":
                              SumO[-1] = int(str(Formula[l-1])+str(Formula[l]))-1 

                       elif Formula[NumIndex-1] == "N":
                              SumN.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-2] == "N":
                              SumN[-1] = int(str(Formula[l-1])+str(Formula[l]))-1 

                       else:
                            pass
                   else:
                       pass
              SumC = [int(x) for x in SumC]
              SumH = [int(x) for x in SumH]
              SumO = [int(x) for x in SumO]
              SumN = [int(x) for x in SumN]

              print("\t SpecieName: {0}".format(Speciename))
              print("\t Formula: {0}".format(Formula))
              print("\t Number of GAVs: {0}".format(NumberOfGAVs))
              print("\t GAVs and Quantity: ")
              GAVs = str(GroupsApp).split(",")
              Quantity = str(QuantyApp).split(",")
              for j in range(len(GAVs)):
                  print("\t {0}\t{1}".format(GAVs[j], Quantity[j]))
              print("\t Number of rotors: {0}".format(RotoryApp))
              print("\t Symmetry number: {0}".format(SymNumApp))
              print("\t", "__" * 46)
		      
              # ..................
              # print(ThermoParser)
              # ..................
              if BooleanRad.lower() in ["no", "n"]:
                  (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                           speciesname=Speciename,
                                                                                                           formula=Formula,
                                                                                                           numberofgroups=NumberOfGAVs,
                                                                                                           groupid=GAVs,
                                                                                                           quantity=Quantity,
                                                                                                           symmetrynumber=SymNumApp,
                                                                                                           numberofrotors=RotoryApp
                                                                                                           ).thermo_props()
              elif BooleanRad.lower() in ["yes", "y"]:
                  (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                           speciesname=Speciename,
                                                                                                           formula=Formula,
                                                                                                           numberofgroups=NumberOfGAVs,
                                                                                                           groupid=GAVs,
                                                                                                           quantity=Quantity,
                                                                                                           symmetrynumber=SymNumApp,
                                                                                                           numberofrotors=RotoryApp
                                                                                                           ).thermo_props_rads()
              # print(Enthalpy,Entropy,Cp300,Cp400,Cp500,Cp600,Cp800,Cp1000,Cp1500)
              print("\t Calculations:\n\t")
              if Cp1500 == 0.0:
                  print("\t Formula H(298K) S(298K) Cp300K Cp400K Cp500K Cp600K Cp800K Cp1000K date")
                  print(("\t " + str(Formula) + " " + str(Enthalpy) + " " + str(Entropy) + " " + str(Cp300) + " " + str(
                      Cp400) + " " + str(Cp500) + " " + str(Cp600) + " " + str(Cp800) + " " + str(
                      Cp1000) + " " + " " + " " + str(d2) + "\n"))
              else:
                  print("\t Formula H(298K) S(298K) Cp300K Cp400K Cp500K Cp600K Cp800K Cp1000K Cp1500K date")
                  print(("\t " + str(Formula) + " " + str(Enthalpy) + " " + str(Entropy) + " " + str(Cp300) + " " + str(
                      Cp400) + " " + str(Cp500) + " " + str(Cp600) + " " + str(Cp800) + " " + str(Cp1000) + " " + str(
                      Cp1500) + " " + str(d2) + "\n"))
              print("\t", ".." * 46)
              # =============================================================
              Temps = [300, 400, 500, 600, 800, 1000, 1500]
              (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                 cp400=Cp400,
                                                 cp500=Cp500,
                                                 cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                 cp1500=Cp1500, temps=Temps).fit_termo()
              print("\t", ".." * 46)
              # =============================================================
              """
              print("\t #C     = ", NumbC)
              print("\t #H     = ", NumbH)
              print("\t #O     = ", NumbO)
              print("\t #N     = ", NumbN)
              """
              NumbC = sum(SumC)
              NumbH = sum(SumH)
              NumbO = sum(SumO)
              NumbN = sum(SumN)

              print_elem_info(NumbC, NumbH, NumbO, NumbN)

              # -------------------------------------------------------------------------------------------
              if  linearity.lower() in ["no","n"]:
                  Cp0linear   = round((3.5) * (1.9), 2)
                  CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
              elif linearity.lower() in ["yes", "y"]:
                  Cp0linear   = round((4.0) * (1.9), 2)
                  CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 2.0) * (1.9), 2)
              NumberAtoms = round((NumbC + NumbH + NumbO + NumbN), 2)
              print("\t Cp0    = ", Cp0linear)
              print("\t CpINF  = ", CpINFlinear)
              print("\t", ".." * 46)

              (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                             GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                             RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)
              # =====================================================
              Doc     = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).doc_format()
              ReRun   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).rerun_format()
              Thermoc = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).therm_format()
              Datac   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7,BP)
              # CountingAtoms = "C2H6"
              SumC.clear()
              SumH.clear()
              SumO.clear()
              SumN.clear()

              # ========================================================
              # UNZIPPING GROUP VALUES FOR SAVE THEM IN WORD FILE...
              # ========================================================
       elif Question1 == 2:
          for k in species_indices_list: 
              print("\t Index line #:\t " + str(k))

              u = species_indices_list.index(k)
              CheckIn = str(InputFile.iloc[k+2]).split()
              GroupsApp = []
              QuantyApp = []

              if  "radical" in CheckIn:
                  #print("\t Radical data")
                  SPECIENAME   = ((InputFile.iloc[k+1])[-1])
                  FORMULA      = ((InputFile.iloc[k+3]).str.split())[-1][-1]
                  PARENT       = ((InputFile.iloc[k+5]).str.split())[-1][-1]
                  NumberOfGAVs = ((InputFile.iloc[k+8]).str.split())[-1][-1]
                  #HeadGAVs     = ((InputFile.iloc[k+10]))
                  BooleanRad   = "YES"
                  GAVsCounter  = 0
                  while GAVsCounter < int(NumberOfGAVs):
                        HeadGAVs    = ((InputFile.iloc[k+10]))
                        if "|" in (HeadGAVs)[-1]:
                           #print("\t Symbol | exists")                  
                           BarSplitted   = (HeadGAVs)[-1].split("|")

                           NumbGAV1  = ((BarSplitted[0]).split("-")[0]).strip()
                           GAVName1  = ((BarSplitted[0]).split("-")[1]).strip()
                           GAVtimes1 = ((BarSplitted[0]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName1)
                           QuantyApp.append(GAVtimes1)

                           NumbGAV2  = ((BarSplitted[1]).split("-")[0]).strip()
                           GAVName2  = ((BarSplitted[1]).split("-")[1]).strip()
                           GAVtimes2 = ((BarSplitted[1]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName2)
                           QuantyApp.append(GAVtimes2)
                           
                           k += 1
                           GAVsCounter += 2
                        else:
                            
                           NumbGAV  = (HeadGAVs)[-1].split("-")[0]
                           GAVName  = (HeadGAVs)[-1].split("-")[1]
                           GAVtimes = (HeadGAVs)[-1].split("-")[-1]
                           GroupsApp.append(GAVName)
                           QuantyApp.append(GAVtimes)
                           
                           k += 1
                           GAVsCounter += 1                  
                  Parent = PARENT.upper()
                  RotoryApp  = InputFile.iloc[species_indices_list2[u]-6]
                  SymNumApp  = InputFile.iloc[species_indices_list2[u]-5]
              elif "molecule" in CheckIn:
                
                  SPECIENAME   = (InputFile.iloc[k+1])[-1]
                  FORMULA      = ((InputFile.iloc[k+3]).str.split())[-1][-1]
                  NumberOfGAVs = ((InputFile.iloc[k+5]).str.split())[-1][-1]
                  BooleanRad = "NO"
                  GAVsCounter = 0
                  while GAVsCounter < int(NumberOfGAVs):
                        HeadGAVs    = ((InputFile.iloc[k+7]))
                        if "|" in (HeadGAVs)[-1]:
                                 
                           BarSplitted   = (HeadGAVs)[-1].split("|")
                           
                           NumbGAV1  = ((BarSplitted[0]).split("-")[0]).strip()
                           GAVName1  = ((BarSplitted[0]).split("-")[1]).strip()
                           GAVtimes1 = ((BarSplitted[0]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName1)
                           QuantyApp.append(GAVtimes1)
                           #print("\t GAV#" + str(NumbGAV1) + "\t " + GAVName1 + " \t " + GAVtimes1)
                           NumbGAV2  = ((BarSplitted[1]).split("-")[0]).strip()
                           GAVName2  = ((BarSplitted[1]).split("-")[1]).strip()
                           GAVtimes2 = ((BarSplitted[1]).split("-")[-1]).strip()
                           GroupsApp.append(GAVName2)
                           QuantyApp.append(GAVtimes2)
                           #print("\t GAV#" + str(NumbGAV2) + "\t " + GAVName2 + " \t " + GAVtimes2)
                           k += 1
                           GAVsCounter += 2
                        else:
                           #print("\t Symbol | not exists")
                           NumbGAV  = (HeadGAVs)[-1].split("-")[0]
                           GAVName  = (HeadGAVs)[-1].split("-")[1]
                           GAVtimes = (HeadGAVs)[-1].split("-")[-1]
                           GroupsApp.append(GAVName)
                           QuantyApp.append(GAVtimes)
                           #print("\t GAV#" + str(NumbGAV) + "\t " + GAVName + " \t " + GAVtimes)
                           k += 1
                           GAVsCounter += 1
                  PARENT     = "NO" 
                  Parent     = PARENT.upper()
                  RotoryApp  = InputFile.iloc[species_indices_list2[u]-3]
                  SymNumApp  = InputFile.iloc[species_indices_list2[u]-2]
              GroupsApp = [x.strip() for x in GroupsApp]
              GroupsApp = ",".join(GroupsApp)
              QuantyApp = [x.strip() for x in QuantyApp]
              QuantyApp = ",".join(QuantyApp)
              # Block added 
              Speciename   = SPECIENAME
              Formula      = FORMULA
              Formula      = str(Formula).upper()
              Formula      = str(Formula).strip()
              NumberOfGAVs = NumberOfGAVs
              GroupsApp    = GroupsApp
              QuantyApp    = QuantyApp
              SymNumApp    = int(SymNumApp[-1][-1])
              RotoryApp    = int(RotoryApp[-1][-1])
              BooleanRad   = BooleanRad.upper()
              
              linearity    = input("\t this molecule is linear? (yes/y or No/n)    ")
              Variable2Test(linearity)
              

              ManyC = Formula.count('C')
              ManyH = Formula.count('H')
              ManyO = Formula.count('O')
              ManyN = Formula.count('N')
              SumC = []
              SumH = []
              SumO = []
              SumN = []
              if ManyC == 0:
                 SumC.append(0)
              else:
                 pass
              if ManyH == 0:
                 SumH.append(0)
              else:
                 pass
              if ManyO == 0:
                 SumO.append(0)
              else:
                 pass
              if ManyN == 0:
                 SumN.append(0)
              else:
                 pass

              for l in range(len(Formula)):
                   if str(Formula[l]).isalpha() == True:
                       if   str(Formula[l]) == "C":
                            SumC.append(1)                      
                       elif str(Formula[l]) == "H":
                            SumH.append(1)
                       elif str(Formula[l]) == "O":
                            SumO.append(1)                      
                       elif str(Formula[l]) == "N":
                            SumN.append(1)
                       else:
                            pass                    
                   elif str(Formula[l]).isnumeric() == True:
                       NumIndex = l

                       if   Formula[NumIndex-1] == "C":
                              SumC.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-1] == "H":
                              #print("This number corresponds to #H")
                              SumH.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-1] == "O":
                              #print("This number corresponds to #O")
                              SumO.append(int(str(Formula[l]))-1)
                       elif Formula[NumIndex-1] == "N":
                              #print("This number corresponds to #N")
                              SumN.append(int(str(Formula[l]))-1)
                       else:
                            pass
                   else:
                       pass
              SumC = [int(x) for x in SumC]
              SumH = [int(x) for x in SumH]
              SumO = [int(x) for x in SumO]
              SumN = [int(x) for x in SumN]
              print("\t SpecieName: {0}".format(Speciename))
              print("\t Formula: {0}".format(Formula))
              # ------------------------------------
              # ************************************
              # INPUT GAVS
              # ************************************
              print("\t Currently GAVs in file:")
              print("\t GAVs and Quantity: ")
              GAVs = str(GroupsApp).split(",")
              Quantity = str(QuantyApp).split(",")
              for j in range(len(GAVs)):
                  print("\t {0}\t{1}".format(GAVs[j], Quantity[j]))
              # While loop to count the GAVs
              NumberOfGAVs2 = Inputs("\t How many extra groups?: (must be an integer) \t").input_number()
              fGAVs2 = []
              Quantity2 = []
              #=====================================
              print("\t Give the group's name + Quantity")
              print("\t Type either C/C/H3 1   or  c/c/h3 1, or simply give the name and press enter; C/C/H3")
              SumaGAVs2 = 0
              counter = 0
              while SumaGAVs2 < NumberOfGAVs2:
                  counter += 1
                  NewGAV = input("\t " + str(counter) + " -  ")
                  Variable2Test(NewGAV)
                  UnGAV = str(str(NewGAV).split(" ")[0])
                  UnQuanto = str(NewGAV).split(" ")[-1]
                  if UnQuanto == UnGAV:
                      UnQuanto = "1"
                  elif len(UnQuanto) == 0:
                      UnQuanto = "1"
                  else:
                      pass
                  fGAVs2.append(UnGAV.upper())
                  Quantity2.append(UnQuanto)
                  SumaGAVs2 = int(SumaGAVs2) + int(UnQuanto)
                  # End while loop
              # Extra loop allowing user to edit GAVs in case of misstyping
              print("\t","--"*46)
              if len(fGAVs2) >1:
                 GroupsApp2 = (",").join(fGAVs2)
                 QuantyApp2 = (",").join(Quantity2)
                 GroupsApp = str(GroupsApp)+","+str(GroupsApp2)
                 QuantyApp = str(QuantyApp)+","+str(QuantyApp2)
              elif len(fGAVs2) ==1:
                 GroupsApp2 = str(fGAVs2[0])
                 QuantyApp2 = str(Quantity2[0])
                 GroupsApp  = str(GroupsApp)+","+str(GroupsApp2)
                 QuantyApp  = str(QuantyApp)+","+str(QuantyApp2)
              else:
                 pass
		      # ******************************
		      # END INPUT GAVS
		      # ******************************
              # -------------------------------------------------
              print("\t Number of GAVs: {0}".format(NumberOfGAVs))
              print("\t GAVs and Quantity: ")
              GAVs = str(GroupsApp).split(",")
              Quantity = str(QuantyApp).split(",")
              for j in range(len(GAVs)):
                  print("\t {0}\t{1}".format(GAVs[j], Quantity[j]))
              print("\t Number of rotors: {0}".format(RotoryApp))
              print("\t Symmetry number: {0}".format(SymNumApp))
              print("\t", "__" * 46)

              if BooleanRad.lower() in ["no","n"]:
                  (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                           speciesname=Speciename,
                                                                                                           formula=Formula,
                                                                                                           numberofgroups=NumberOfGAVs,
                                                                                                           groupid=GAVs,
                                                                                                           quantity=Quantity,
                                                                                                           symmetrynumber=SymNumApp,
                                                                                                           numberofrotors=RotoryApp
                                                                                                           ).thermo_props()
              elif BooleanRad.lower() in ["yes", "y"]:
                  (Enthalpy, Entropy, ZIPTest, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500) = Thermo(dfObj,
                                                                                                           speciesname=Speciename,
                                                                                                           formula=Formula,
                                                                                                           numberofgroups=NumberOfGAVs,
                                                                                                           groupid=GAVs,
                                                                                                           quantity=Quantity,
                                                                                                           symmetrynumber=SymNumApp,
                                                                                                           numberofrotors=RotoryApp
                                                                                                           ).thermo_props_rads()
              print("\t Calculations:\n\t")
              if Cp1500 == 0.0:
                  print("\t Formula H(298K) S(298K) Cp300K Cp400K Cp500K Cp600K Cp800K Cp1000K date")
                  print(("\t " + str(Formula) + " " + str(Enthalpy) + " " + str(Entropy) + " " + str(Cp300) + " " + str(
                      Cp400) + " " + str(Cp500) + " " + str(Cp600) + " " + str(Cp800) + " " + str(
                      Cp1000) + " " + " " + " " + str(d2) + "\n"))
              else:
                  print("\t Formula H(298K) S(298K) Cp300K Cp400K Cp500K Cp600K Cp800K Cp1000K Cp1500K date")
                  print(("\t " + str(Formula) + " " + str(Enthalpy) + " " + str(Entropy) + " " + str(Cp300) + " " + str(
                      Cp400) + " " + str(Cp500) + " " + str(Cp600) + " " + str(Cp800) + " " + str(Cp1000) + " " + str(
                      Cp1500) + " " + str(d2) + "\n"))
              print("\t", ".." * 46)
              # =============================================================
              Temps = [300, 400, 500, 600, 800, 1000, 1500]
              (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GroupFiles, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                 cp400=Cp400,
                                                 cp500=Cp500,
                                                 cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                 cp1500=Cp1500, temps=Temps).fit_termo()
              print("\t", ".." * 46)
              # =============================================================
              NumbC = sum(SumC)
              NumbH = sum(SumH)
              NumbO = sum(SumO)
              NumbN = sum(SumN)

              print_elem_info(NumbC, NumbH, NumbO, NumbN)

              if  linearity.lower() in ["no","n"]:
                  Cp0linear   = round((3.5) * (1.9), 2)
                  CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
                  NumberAtoms = round((NumbC + NumbH + NumbO + NumbN), 2)
                  print("\t Cp0    = ", Cp0linear)
                  print("\t CpINF  = ", CpINFlinear)
                  print("\t", ".." * 46)
              elif linearity.lower() in ["yes", "y"]:
                  Cp0linear   = round((4.0) * (1.9), 2)
                  CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 2.0) * (1.9), 2)
                  NumberAtoms = round((NumbC + NumbH + NumbO + NumbN), 2)
                  print("\t Cp0    = ", Cp0linear)
                  print("\t CpINF  = ", CpINFlinear)
                  print("\t", ".." * 46)
              # ======================================================
              (b1, b2, b3, b4, b5, b6, b7) = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                             GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                             RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)
              # =====================================================
              Doc     = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).doc_format()
              ReRun   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).rerun_format()
              Thermoc = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).therm_format()
              Datac   = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad,
                                 GAVs, Quantity, Formula, Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                 RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC, NumbH, NumbO, NumbN, CpINFlinear, linearity).dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7)
              # CountingAtoms = "C2H6"
              SumC.clear()
              SumH.clear()
              SumO.clear()
              SumN.clear()

       # ========================================================
       z.write("END")
       z.close()
       y.close()
       f.close()
       t.close()
       q.close()
       #
              # ========================================================
              # UNZIPPING GROUP VALUES FOR SAVE THEM IN WORD FILE...
              # ========================================================
       #SPECIESlist = []
       """       
       Speciename   = InputFile.iloc[k, 0]
       Formula      = InputFile.iloc[k, 1]
       Formula      = str(Formula).upper()
       NumberOfGAVs = InputFile.iloc[k, 2]
       GroupsApp    = InputFile.iloc[k, 3]
       QuantyApp    = InputFile.iloc[k, 4]
       SymNumApp    = InputFile.iloc[k, 5]
       RotoryApp    = InputFile.iloc[k, 6]
       BooleanRad   = str(InputFile.iloc[k, 7]).upper()
       pparent      = InputFile.iloc[k, 8]
       NumbC        = InputFile.iloc[k, 9]
       NumbH        = InputFile.iloc[k, 10]
       NumbO        = InputFile.iloc[k, 11]
       Parent       = pparent.upper()
       """
	   
    # *************************************************************
    # Switching to Plotter mode
    # =============================================================
elif MainMenu == 4:
    #
    print("\t Plotting thermochemistry...")
    print("\t ","--" * 46)
    print("\t List of files available to plot thermochemistry properties are:")
    print("\t","==" * 46)
    for u in range(len(THERMFiles)):
        print("\t", str(THERMFiles[u]).split(psep)[-1])
    print("\t","==" * 46)
    print("\t Choose one and insert its name as requested. However, if your file is not listed")
    print("\t please copy and paste it into '/Therm23/OutputsDir/File2Plot.dat' directory")
    print("\t file should have an extension 'dat' and it must have 2 set of NASA polynomials format")
    print("\t",".." * 46)
    print("\t Would you like single plots or to compare thermo files?")
    print("\t For single plots type: 1 \t\t\t(or just press the 'Enter' key)")
    print("\t To compare plots type: 2")

    while True:
        try:
            TypeOfPlotter = Inputs("\t ").plot_switcher() or 0
            print("\t Max temperature to plot in K: (integer, i.e. 3000 or 5000)")
            try:
               MaxT = Inputs("\t ").input_number()
            except:
               print("\t User failed in providing a Mac Temp to plot, 3000 is the default value to be used then")
               MaxT = 3000
            if   MaxT <= 1500:
                 print("\t Max Temp <= 1500 is not allowed... taking default value for 3000")
                 MaxT = 3000
            elif MaxT > 1500:
                 pass

            if  TypeOfPlotter == 1:
                print("\t" + ".."*20)
                print("\t Running plotter in individual mode")
                print("\t" + ".."*20)
                NumberFiles = Inputs("\t How many files would you like to plot?:\t").input_number()
                Thermo2Plot = []
                if NumberFiles == 0:
                    print("\t 0 files provided.")
                    pass
                else:
                    print("\t Please provide the file name(s)")
                    for y in range(NumberFiles):
                        while True:
                            try:
                                FileName1  = input("\t Please give file's name #" + str(y+1) + ":\t ")
                                PandasFile = pd.read_table(CWDPath + "/OutputsDir/" + str(FileName1))
                                Thermo2Plot.append(FileName1)
                                break
                            except:
                                print("\t> " + str(FileName1) + " file not in the list!")
                                continue
                    for u in Thermo2Plot:
                        ExtractorPolys = CoeffReader(u,MaxT).reading_coeffs()
                break
            elif TypeOfPlotter == 2: #["multi", "MULTI", "Multi", "MUlti", "MULti", "MULTi", "mULTI", "muLTI", "mulTI", "multI"]:
                print("\t" + ".."*20)
                print("\t Running plotter in Multi mode")
                print("\t" + ".."*20)
                print("\t This mode allows the user to compare thermochemistry properties making graphs with them")
                print("\t from two different thermo files.")
                NumberFiles = 2 #Inputs("\t How many files would you like to plot?:\t").input_number()
                Thermo2Plot = []
                print("\t Please type the two file names to compare from the list above:")
                for y in range(NumberFiles):
                    while True:
                        try:
                            FileName1  = input("\t Please give file's name #" + str(y+1) + ":\t ")
                            PandasFile = pd.read_table(CWDPath + "/OutputsDir/" + str(FileName1))
                            Thermo2Plot.append(FileName1)
                            break
                        except:
                            print("\t> " + str(FileName1) + " file not in the list!")
                            continue
                ExtractorPolys = CoeffReader(Thermo2Plot[0], MaxT).MultiPlotter(Thermo2Plot)
                break
            else:
                print("\t Wrong answer, try again... ")
                continue
        except:
                #print("\t Wrong answer, try again... ")
                break

    # *************************************************************
    # Switching to Heat Capacities fitter mode
    # This is a mode that allows user to fill gaps in
    # heat capacities values in a specific file
    # =============================================================
elif MainMenu == 5:
    print("\t","--" * 46)
    print("\t Missing data (CPs) fitter type 1")
    print("\t Fit GAVs (CPs) and get polynomial sets type 2")
    print("\t Type 0 to exit.")
    # print("\t Now code is looking for files with extension *.miss and listing next:")

    # Getting the path where we are running this code (to try to find *.miss files)
    CWD = os.getcwd()
    Answr = Inputs("\t ").plot_switcher() or 0
    if   Answr == 0:
         print("\t Option are '1 or 2', closing program now... \n\t Please run again the code and type a right answer.")
         sys.exit(1)
    elif Answr == 1:
         print("\t" + ".."*20)
         print("\t Running missing data (CPs) fitter mode")
         print("\t" + ".."*20)
         ListOFMissFiles = glob.glob(CWD + "/OutputsDir/*.miss")
         print("\t Code is automatically looking for *.miss files and converting (calculating missed data) in a *.NoMiss")
         for j in ListOFMissFiles:
            print("\t Processing... " + str(j))
            #try:
            PandaFile = Parser([j]).data()
            #except:
            #    PandaFile = Parser([j]).data2()

            # print(PandaFile.iloc[0, :])
            SpeciesNames1 = []
            Hentalpies    = []
            Entropies     = []
            Heat300       = []
            Heat400       = []
            Heat500       = []
            Heat600       = []
            Heat800       = []
            Heat1000      = []
            Heat1500      = []

            for q in range(len(PandaFile)):
                CP300  = float(PandaFile.iloc[q, 3])
                CP400  = float(PandaFile.iloc[q, 4])
                CP500  = float(PandaFile.iloc[q, 5])
                CP600  = float(PandaFile.iloc[q, 6])
                CP800  = float(PandaFile.iloc[q, 7])
                CP1000 = float(PandaFile.iloc[q, 8])
                CP1500 = float(PandaFile.iloc[q, 9])
                SpeciesNames1.append(PandaFile.iloc[q, 0])
                Hentalpies.append(PandaFile.iloc[q, 1])
                Entropies.append(PandaFile.iloc[q, 2])
                Heat1500.append(CP1500)

                Tempos = [300, 400, 500, 600, 800, 1000, 1500]
                A1, A2, A3, A4 = Parser(groupfiles=GroupFiles, cp300=CP300, cp400=CP400, cp500=CP500, cp600=CP600,
                                        cp800=CP800, cp1000=CP1000, cp1500=CP1500, temps=Tempos).guess_Cps()

                def funcCPs(T, a1, a2, a3, a4):
                    return (a1 + a2 * (T) + a3 * (T) ** 2 + a4 * (T) ** 3)
                papa300 = (round(funcCPs(300, A1, A2, A3, A4), 2))
                Heat300.append(papa300)
                papa400 = (round(funcCPs(400, A1, A2, A3, A4), 2))
                Heat400.append(papa400)
                papa500 = (round(funcCPs(500, A1, A2, A3, A4), 2))
                Heat500.append(papa500)
                papa600 = (round(funcCPs(600, A1, A2, A3, A4), 2))
                Heat600.append(papa600)
                papa800 = (round(funcCPs(800, A1, A2, A3, A4), 2))
                Heat800.append(papa800)
                papa1000 = (round(funcCPs(1000, A1, A2, A3, A4), 2))
                Heat1000.append(papa1000)

            formatter = lambda x : np.format_float_positional(np.float64(x), unique=False, precision=2)
            
            Hentalpies    = [formatter(x) for x in Hentalpies]
            Entropies     = [formatter(x) for x in Entropies]
            Heat300       = [formatter(x) for x in Heat300]
            Heat400       = [formatter(x) for x in Heat400]
            Heat500       = [formatter(x) for x in Heat500]
            Heat600       = [formatter(x) for x in Heat600]
            Heat800       = [formatter(x) for x in Heat800]
            Heat1000      = [formatter(x) for x in Heat1000]
            Heat1500      = [formatter(x) for x in Heat1500]

            d = {"SPECIES": SpeciesNames1, "Hf": Hentalpies, "S": Entropies, "Cp 300": Heat300,
                " 400 ": Heat400, " 500 ": Heat500, " 600 ": Heat600, " 800 ": Heat800, " 1000 ": Heat1000,
                " 1500 ": Heat1500}
            Split0 = j.rsplit(psep,1)[0]
            Split1 = j.split(psep)[-1]
            Split2 = Split1.split(".miss")[0]
            Outputfile = pd.DataFrame(data=d)
            Outputfile.to_csv(Split0 + psep + Split2 + ".NoMiss", sep="\t", index=False)
            print("\t ","--" * 46)
            print("\t " + str(Split2) + ".NoMiss" + " created.")
            # print(PandaFile)
            print("\t ","--" * 46)
    elif Answr == 2:
        print("\t" + ".."*20)
        print("\t Running GAVs (CPs) fitter mode")
        print("\t" + ".."*20)
        #print("\t For interactive mode type 1")
        #print("\t For automatic mode type 2" + " "*20 + "(you may need a LST file for this)")
        Answr2 = 2 #Inputs("\t ").plot_switcher() or 0
        if  Answr2 == 0:
            print("\t Option typed '0', closing program now...")
            sys.exit(1)
        elif Answr2 == 1:
            pass
        elif Answr2 == 2:
            print("\t","--" * 46)
            print("\t List of LST files available:")
            print("\t","==" * 46)
            ListOfReRuns = []
            for u in range(len(LSTinputs)):
                print("\t", str(LSTinputs[u]).split(psep)[-1])
            print("\t","==" * 46)
            # loop for check if file exist in list
            NumberFiles = Inputs("\t How many files would you like to plot?:\t").input_number() or 0

            Thermo2Plot = []
            if NumberFiles == 0:
                print("\t 0 files provided.")
                pass
            elif NumberFiles == "all":
                for y in LSTinputs:
                    Thermo2Plot.append(y)
            else:
                for y in range(NumberFiles):
                    while True:
                        try:
                            FileName1  = input("\t Please give file's name #" + str(y+1) + ":\t ")
                            FileName1  = FileName1.strip()
                            PandasFile = pd.read_csv(CWDPath + "/OutputsDir/" + str(FileName1))
                            Thermo2Plot.append(CWDPath + "/OutputsDir/" + str(FileName1))
                            break
                        except:
                            print("\t> " + str(FileName1) + " file not in the list!")
                            continue
            print("\t Code is working")
            # saving on file
            DatFile      = FileName1 + ".dat"
            DatFileOut   = OUTPUT + psep + DatFile
            z = open(DatFileOut, "w")
            z.write("THERMO\n")
            z.write("   300.00  1000.00  5000.00\n")
            for m in Thermo2Plot:
                print("\t","**"*46)
                print("\t File:")
                mnm = str(m).split(psep)[-1]
                print("\t " + mnm)
                PdFile = Parser([m]).data3()
                print("\t ", PdFile)                 
                print("\t ", PdFile.iloc[0,:].to_list() )                 
                print("\t ", PdFile.iloc[0,0] )                 
                print("\t ", PdFile.iloc[0,0].split() )                 
                print("\t HOLA")                 
                for k in range(len(PdFile)):
                    for p in range(2):
                        GAV       = PdFile.iloc[k,0].split()[0]        ; print(GAV)
                        Enthalpy  = float(PdFile.iloc[k,0].split()[1]) ; print(Enthalpy)
                        Entropy   = float(PdFile.iloc[k,0].split()[2]) ; print(Entropy)
                        Cp300     = float(PdFile.iloc[k,0].split()[3]) ; print(Cp300)
                        Cp400     = float(PdFile.iloc[k,0].split()[4]) ; print(Cp400)
                        Cp500     = float(PdFile.iloc[k,0].split()[5]) ; print(Cp500)
                        Cp600     = float(PdFile.iloc[k,0].split()[6]) ; print(Cp600)
                        Cp800     = float(PdFile.iloc[k,0].split()[7]) ; print(Cp800)
                        Cp1000    = float(PdFile.iloc[k,0].split()[8]) ; print(Cp1000)
                        Cp1500    = float(PdFile.iloc[k,0].split()[9]) ; print(Cp1500)
                        RotoryApp = int(PdFile.iloc[k,0].split()[-1])  ; print(RotoryApp)
                        try:
                            NumbC     = int(PdFile.iloc[k,0].split()[11])  ; print(NumbC)
                            NumbH     = int(PdFile.iloc[k,0].split()[12])  ; print(NumbH)
                            NumbO     = int(PdFile.iloc[k,0].split()[13])  ; print(NumbO)
                            NumbN     = int(PdFile.iloc[k,0].split()[14])  ; print(NumbN)
                        except:
                            NumbC     = float(PdFile.iloc[k,0].split()[11])  ; print(NumbC)
                            NumbH     = float(PdFile.iloc[k,0].split()[12])  ; print(NumbH)
                            NumbO     = float(PdFile.iloc[k,0].split()[13])  ; print(NumbO)
                            NumbN     = float(PdFile.iloc[k,0].split()[14])  ; print(NumbN)
                        #
                        Temps     = [300, 400, 500, 600, 800, 1000, 1500]
                        (a1, a2, a3, a4, a5, a6, a7) = Parser(groupfiles=GAV, h=Enthalpy, s=Entropy, cp300=Cp300,
                                                      cp400=Cp400,
                                                      cp500=Cp500,
                                                      cp600=Cp600, cp800=Cp800, cp1000=Cp1000,
                                                      cp1500=Cp1500, temps=Temps).fit_termo()
                        # Converting data from NASA to thermochemistry: Cps/R, H and S to plot
                        CPs1  = []
                        Hs1   = []
                        Ss1   = []
                        CPs2  = []
                        CPs2b = []
                        Hs2   = []
                        Hs2b  = []
                        Ss2   = []
                        Ss2b  = []
                        # BEGINNING OF FUNCTIONS
                        def funcCP(T, m1, m2, m3, m4, m5):
                            return ((m1) + (m2 * (T)) + (m3 * (T) ** 2) + (m4 * (T) ** 3) + (m5 * (T) ** 4))*R
                        def funcH(T, m1, m2, m3, m4, m5, m6):
                            T2 = T * T
                            T4 = T2 * T2
                            return (m1 + m2 * T / 2 + m3 * T2 / 3 + m4 * T2 * T / 4 + m5 * T4 / 5 + m6 / T) * R * T / 1000 #kcal unit
                        def funcS(T, m1, m2, m3, m4, m5, m7):
                            import math
                            T2 = T * T
                            T4 = T2 * T2
                            return (m1 * math.log(T) + m2 * T + m3 * T2 / 2 + m4 * T2 * T / 3 + m5 * T4 / 4 + m7) * R                        
                        # ..................
                        Cp0linear   = round((3.5) * (1.9), 2)
                        CpINFlinear = round((3 * (NumbC + NumbH + NumbO + NumbN) - 1.5) * (1.9), 2)
                        # END OF FUNCTIONS
                        # -----------------------------------------------------
                        Temperatures2Plot1 = np.arange(300, 1100, 100).tolist()
                        Temperatures2Plot2 = np.arange(1000, 5100, 100).tolist()
                        CWD2 = os.getcwd()
                        OutputPlotPath = CWD2 + "/OutputsDir"
                        if not os.path.exists(OutputPlotPath + "/Plots/Fitted/" + mnm):
                               os.makedirs(OutputPlotPath + "/Plots/Fitted/" + mnm)
                        for d1 in (Temperatures2Plot1):
                            CPs1.append(funcCP(d1, float(a1), float(a2), float(a3), float(a4), float(a5)))
                            Hs1.append(funcH(d1, float(a1), float(a2), float(a3), float(a4), float(a5), float(a6)))
                            Ss1.append(funcH(d1, float(a1), float(a2), float(a3), float(a4), float(a5), float(a7)))
                        for d2 in (Temperatures2Plot2):
                            CPs2.append(funcCP(d2, float(a1), float(a2), float(a3), float(a4), float(a5)))
                            Hs2.append(funcH(d2, float(a1), float(a2), float(a3), float(a4), float(a5), float(a6)))
                            Ss2.append(funcH(d2, float(a1), float(a2), float(a3), float(a4), float(a5), float(a7)))                            
                        # ----------
                        (b1, b2, b3, b4, b5, b6, b7, BP) = Outputs("NamesFilesDoc", "NamesFilesTherm", "NamesFilesInp", "ThermoFiles", GAV, "Parent", "BooleanRad",
                                                               "GAVs", "Quantity", "Formula", Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                                               RotoryApp, "SymNumApp", d22, "NumberOfGAVs", NumbC, NumbH, NumbO, NumbN, CpINFlinear, "NO").ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)
                        #
                        #(b1, b2, b3, b4, b5, b6, b7, BP) = Outputs(NamesFilesDoc, NamesFilesTherm, NamesFilesInp, ThermoFiles, Speciename, Parent, BooleanRad, GAVs, Quantity, Formula, 
                        #                           Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500, RotoryApp, SymNumApp, d2, NumberOfGAVs, NumbC,
                        #                           NumbH, NumbO, NumbN, CpINFlinear, Linearity).ExtraWilhoit(a1, a2, a3, a4, a5, a6, a7)
                        # Saving coefficients to file
                        #BP = 1000
                        if p == 0:
                            if isinstance(NumbC, float):
                               NumbC = round(NumbC,1)
                               print(NumbC)
                            else:
                               pass
                            if isinstance(NumbH, float):
                               NumbH = round(NumbH,1)
                            else:
                               pass
                            if isinstance(NumbO, float):
                               NumbO = round(NumbO,1)
                            else:
                               pass
                            if isinstance(NumbN, float):
                               NumbN = round(NumbN,1)
                            else:
                               pass
                            #
                            Datac   = Outputs("DocFileOut", "DatFileOut", "LSTFileOut", "RerunFileOut", GAV, "Parent", "BooleanRad",
                                              "GAVs", "Quantity", "Formula", Enthalpy, Entropy, Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500,
                                              RotoryApp, "SymNumApp", d22, "NumberOfGAVs", NumbC, NumbH, NumbO, NumbN, CpINFlinear, "NO").dat_format(a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, BP)
                        else:
                            pass
                        # =============
                        # end loop
                        # =============
            z.write("END")
            z.close()
        else:
            pass
    else:
        sys.exit(1)
    print("\t","==" * 46)
	   
    # *************************************************************
    # Switching to LST mode
    # =============================================================
else:
    print("\t User asked for termination of the program.")
    sys.exit(1)

"""
print("\t Note: Please find your thermo data storaged in Therm23.date.therm")
print("\t Log file with all calculations requested could be find in Therm23.date.doc")
print("\t & the Thermo calculations requested in Polynomial format can be find in Therm23.date.dat")
print("."*50)
print("Thanks for using Therm23 python C3 code")
print("'La Verdad Os Hara Libres'")
"""
