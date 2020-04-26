import sys
import numpy as np

def trilaterate3D(distance):
    p1=np.array(distance[0][:3])
    p2=np.array(distance[1][:3])
    p3=np.array(distance[2][:3])       
    p4=np.array(distance[3][:3])
    r1=distance.iloc[0,-1] / 2
    r2=distance.iloc[1,-1] / 2
    r3=distance.iloc[2,-1] / 2
    r4=distance.iloc[3,-1] / 2
    e_x=(p2-p1)/np.linalg.norm(p2-p1)
    i=np.dot(e_x,(p3-p1))
    e_y=(p3-p1-(i*e_x))/(np.linalg.norm(p3-p1-(i*e_x)))
    e_z=np.cross(e_x,e_y)
    d=np.linalg.norm(p2-p1)
    j=np.dot(e_y,(p3-p1))
    x=((r1**2)-(r2**2)+(d**2))/(2*d)
    y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
    z1=np.sqrt(abs(r1**2-x**2-y**2))
    z2=np.sqrt(r1**2-x**2-y**2)*(-1)
    ans1=p1+(x*e_x)+(y*e_y)+(z1*e_z)
    ans2=p1+(x*e_x)+(y*e_y)+(z2*e_z)
    dist1=np.linalg.norm(p4-ans1)
    dist2=np.linalg.norm(p4-ans2)
    if np.abs(r4-dist1)<np.abs(r4-dist2):
        return ans1
    else: 
        return ans2

if __name__ == "__main__":
    '''
    # Retrive file name for input data
    if(len(sys.argv) == 1):
        print("Please enter data file name.")
        exit()
    
    filename = sys.argv[1]

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    distances = []
    for line in range(0, len(lines)):
        distances.append(map(float, lines[line].split(' ')))
'''
    # Print out the data
    print ("The input four points and distances, in the format of [x, y, z, d], are:")
    for p in range(0, len(distance)):
        print(distance[p])

    # Call the function and compute the location 
    location = trilaterate3D(distance)
    print 
    print ("The location of the point is: " + str(location))