import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

##################################################################
# 
##################################################################
class Scalar3d:
  def __init__(self,Filename):
    print 'Reading from file \''+Filename+'\''
    File = open(Filename,'r')
    # Skip first for lines
    for i in range(4):
      File.readline()
    # number of x- grid cells
    s = File.readline()
    self.nx = int(s.split()[2])
    # number of y- grid cells
    s = File.readline()
    self.ny = int(s.split()[2])
    # number of z- grid cells
    s = File.readline()
    self.nz = int(s.split()[2])
    # x- grid size
    s = File.readline()
    self.Lx = float(s.split()[2])
    # y- grid size
    s = File.readline()
    self.Ly = float(s.split()[2])
    # z- grid size
    s = File.readline()
    self.Lz = float(s.split()[2])
    print ' nx = '+str(self.nx)
    print ' ny = '+str(self.ny)
    print ' nz = '+str(self.nz)
    print ' Lx = '+str(self.Lx)
    print ' Ly = '+str(self.Ly)
    print ' Lz = '+str(self.Lz)
    # Read data
    self.C = np.zeros((self.nx,self.ny,self.nz))
    self.X = np.zeros((self.nx+1,self.ny+1))
    self.Y = np.zeros((self.nx+1,self.ny+1))
    for ix in range(self.nx):
      for iy in range(self.nx):
        for iz in range(self.nz):
          self.C[ix,iy,iz] = float(File.readline())
    for ix in range(self.nx+1):
      for iy in range(self.ny+1):
        self.X[ix,iy] = self.Lx/self.nx*(ix)
        self.Y[ix,iy] = self.Ly/self.ny*(iy)
    File.close()

##################################################################
#  Plot a specified vertical level from a scalar3d data field
##################################################################
  def PlotLevel(self,Level,outFilename):
    C = self.C[:,:,Level]
    ax = plt.gca()
    ax.set_xlim(-self.Lx/self.nx,self.Lx+self.Lx/self.nx)
    ax.set_ylim(-self.Ly/self.ny,self.Ly+self.Ly/self.ny)
    ax.set_aspect('equal')
    p = plt.pcolor(self.X,self.Y,C)
    plt.colorbar(p)
    plt.savefig(outFilename)

def Main(Filename,Level,outFilename):
  phi = Scalar3d(Filename)
  phi.PlotLevel(Level,outFilename)

if (__name__ == '__main__'):
  if (len(sys.argv) != 4):
    print 'Usage: python '+sys.argv[0]+' <filename> <level> <outfilename>'
    sys.exit(0)
  Filename = sys.argv[1]
  Level = sys.argv[2]
  outFilename = sys.argv[3]
  Main(Filename,Level,outFilename)
  
