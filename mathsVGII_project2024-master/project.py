import customtkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math
import numpy as np

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


def rotMat2Eaa(R):
    '''
    Returns the principal axis and angle encoded in the rotation matrix R
    '''
    # Calcular la traza de la matriz
    T = np.trace(R)  # Suma de elementos diagonales

    # Calcular el ángulo a partir de la traza
    angle = np.arccos((T - 1) / 2)

    # Evitar divisiones por 0 cuando el ángulo es cercano a 0
    if np.isclose(angle, 0, atol=1e-8):
        # Sin rotación
        return np.array([0, 0, 0]), 0

    # Calcular la matriz antisimetralizada
    anti_symmetric_part = (R - R.T) / (2 * np.sin(angle))

    # Extraer el eje de rotación
    axis = np.array([
        anti_symmetric_part[2, 1],
        anti_symmetric_part[0, 2],
        anti_symmetric_part[1, 0]
    ])

    return axis, angle


def eAngles2rotM(yaw,pitch,roll): #psi, theta, phi
    '''
    Given a set of Euler angles returns the rotation matrix R
    '''
    R_x = np.array([[1,         0,                  0            ],
                    [0,         math.cos(roll), math.sin(roll) ],
                    [0,         -math.sin(roll), math.cos(roll)  ]
                    ])
 
    R_y = np.array([[math.cos(pitch),    0,      -math.sin(pitch)  ],
                    [0,                     1,      0             ],
                    [math.sin(pitch),   0,      math.cos(pitch)  ]])
 
    R_z = np.array([[math.cos(yaw),    math.sin(yaw),    0],
                    [-math.sin(yaw),    math.cos(yaw),     0],
                    [0,                     0,            1]])
    
 
    R = R_z@R_y@R_x 
    
    return R

def rotM2eAngles(R): #psi, theta, phi
    '''
    Given a rotation matrix R returns a set of Euler angles 
    '''
    if R[2,0]==1:
      pitch = math.asin(R[2][0])
      roll = 0
      yaw = -R[1,1]
    elif R[2,0]==-1:
      pitch = math.asin(R[2][0])
      roll = 0
      yaw = R[1,1]
    else:
      pitch = math.asin(R[2][0]) 
      roll = -math.atan2(-R[2,1]/math.cos(pitch) , R[2,2]/math.cos(pitch) )
      yaw = -math.atan2(-R[1,0]/math.cos(pitch), R[0,0]/math.cos(pitch) )

    return (yaw, pitch, roll)


def rot_vecToRotM(rot_vec):    

    # Convertir ángulos de grados a radianes
    magnitud= magnitude(rot_vec)
    axis=rot_vec/magnitud

    angle=magnitud

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    x=axis[0]
    print("x: ",x)
    y=axis[1]
    z=axis[2]
    x2 = x * x
    y2 = y * y
    z2 = z * z

    R= np.array([[t * x2 + c, t * x * y - s * z, t * x * z + s * y],
                 [t * x * y + s * z, t * y2 + c, t * y * z - s * x],
                 [t * x * z - s * y, t * y * z + s * x, t * z2 + c]])

    
    
    print(R)
    return R

def Eaa2rotM(angle, axis):
    '''
    Returns the rotation matrix R able to rotate vectors an angle 'angle' (in rads) about the axis 'axis'
    '''
    magnitud= magnitude(axis)
    axis=axis/magnitud
    
    # I= np.array([[1,0,0],
    #             [0,1,0],
    #             [0,0,1]])
    # ux = np.array([[0, -axis[2,0], axis[1,0]], [axis[2,0], 0, -axis[0,0]], [-axis[1,0], axis[0,0], 0]])

    # R= I*math.cos(angle)+(I-math.cos(angle))*(axis@np.transpose(axis))+ux*math.sin(angle)

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    x=axis[0]
    y=axis[1]
    z=axis[2]
    x2 = x * x
    y2 = y * y
    z2 = z * z

    R= np.array([[t * x2 + c, t * x * y - s * z, t * x * z + s * y],
                 [t * x * y + s * z, t * y2 + c, t * y * z - s * x],
                 [t * x * z - s * y, t * y * z + s * x, t * z2 + c]])
    

    

    return R

def quaternion_rotation_matrix(Q):

    # Q = Q/np.linalg.norm(Q)
    # print(Q)
    # print(np.linalg.norm(Q))

    #Normalitzem el cuaternio
    magnitud = math.sqrt(Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])
    Q= [Q[i]/magnitud for i in range(4)]

    # Agafem els valors de Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # Primera fila de la matriu
    r00 = q0**2 + q1**2 - q2**2 - q3**2
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Segona fila de la matriu
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = q0**2 - q1**2 + q2**2 - q3**2
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Tercera fila de la matriu
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = q0**2 - q1**2 - q2**2 + q3**2
    
    # 3x3 rotation matrix
    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]])

    return R

def quaternion_multiply(Q0,Q1):
 
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
        
    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
        
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
        
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion
def fromtwovectors(m0, m1):

    #Metodo 1
    # w = np.cross(np.transpose(m0), np.transpose(m1))
    # q = np.array([np.dot(m0,m1), w[0], w[1], w[2]])
    # q[1] += len(q)
    # q[2] += len(q)
    # q[3] += len(q)


    w = np.cross(m0, m1)
    q = np.array([1 + np.dot(m0, m1), w[0], w[1], w[2]])
    
    
    #Metodo 2
    # angle=math.acos(np.dot(m1,np.transpose(m0))/(magnitude(m0)*magnitude(m1)))
    # print("angle: ",angle)
    # q0=np.cos(angle/2)
    # qx=np.sin(angle/2)*(np.cross(np.transpose(m0),np.transpose(m1))/(magnitude(np.cross(m0,m1))))
    # print("qx: ",qx)
    # q1=qx[0]
    # q2=qx[1]
    # q3=qx[2]
    # q=np.array([q0,q1,q2,q3])

    #Metodo 3
    # q = np.append(np.cross(m0, m1), np.dot(m0, m1))

    return q

def magnitude(vector): 
 return np.sqrt(sum(pow(element, 2) for element in vector))

def creacvector_m(x,y):
    x2=x*x
    print("x2: ",x2)
    y2=y*y
    print("y2: ",y2)
    r2 = 4
    vex=np.array([y,r2/(2*np.sqrt(x2+y2)),-x])
    vexmodule= magnitude(vex)
    if x2+y2<r2/2:
        m= np.array([y,abs(np.sqrt(r2-x2-y2)),-x])
    if x2+y2>=r2/2:
        m=(np.sqrt(r2))*vex/vexmodule
    return m

class Arcball(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        # Orientation vars. Initialized to represent 0 rotation
        self.quat = np.array([[1],[0],[0],[0]])
        self.rotM = np.eye(3)
        self.AA = {"axis": np.array([[0],[0],[0]]), "angle":0.0}
        self.rotv = np.array([[0],[0],[0]])
        self.euler = np.array([[0],[0],[0]])

        # configure window
        self.title("Holroyd's arcball")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)

        self.grid_columnconfigure((0,1), weight=0   )
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Cube plot
        self.init_cube()
        self.m0=np.array([0,0,0])
        self.m1=np.array([0,0,0])
        self.canvas = FigureCanvasTkAgg(self.fig, self)  # A tk.DrawingArea.
        self.bm = BlitManager(self.canvas,[self.facesObj])
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.pressed = False #Bool to bypass the information that mouse is clicked
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        
        # Reset button
        self.resetbutton = customtkinter.CTkButton(self, text="Reset", command=self.resetbutton_pressed)
        self.resetbutton.grid(row=3, column=0, padx=(0, 0), pady=(5, 20), sticky="ns")
        
        # Selectable atti
        self.tabview = customtkinter.CTkTabview(self, width=150, height=150)
        self.tabview.grid(row=0, column=1, padx=(0, 20), pady=(20, 0), sticky="nsew")
        self.tabview.add("Axis angle")
        self.tabview.add("Rotation vector")
        self.tabview.add("Euler angles")
        self.tabview.add("Quaternion")

        # Selectable atti: AA
        self.tabview.tab("Axis angle").grid_columnconfigure(0, weight=0)  # configure grid of individual tabs
        self.tabview.tab("Axis angle").grid_columnconfigure(1, weight=0)  # configure grid of individual tabs

        self.label_AA_axis= customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Axis:")
        self.label_AA_axis.grid(row=0, column=0, rowspan=3, padx=(80,0), pady=(45,0), sticky="e")

        self.entry_AA_ax1 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax1.insert(0,"1.0")
        self.entry_AA_ax1.grid(row=0, column=1, padx=(5, 0), pady=(50, 0), sticky="ew")

        self.entry_AA_ax2 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax2.insert(0,"0.0")
        self.entry_AA_ax2.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky="ew")

        self.entry_AA_ax3 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax3.insert(0,"0.0")
        self.entry_AA_ax3.grid(row=2, column=1, padx=(5, 0), pady=(5, 10), sticky="ew")

        self.label_AA_angle = customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Angle:")
        self.label_AA_angle.grid(row=3, column=0, padx=(120,0), pady=(10, 20),sticky="w")
        self.entry_AA_angle = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_angle.insert(0,"0.0")
        self.entry_AA_angle.grid(row=3, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        self.button_AA = customtkinter.CTkButton(self.tabview.tab("Axis angle"), text="Apply", command=self.apply_AA, width=180)
        self.button_AA.grid(row=5, column=0, columnspan=2, padx=(0, 0), pady=(5, 0), sticky="e")

        # Selectable atti: rotV
        self.tabview.tab("Rotation vector").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Rotation vector").grid_columnconfigure(1, weight=0)
        
        self.label_rotV= customtkinter.CTkLabel(self.tabview.tab("Rotation vector"), text="rot. Vector:")
        self.label_rotV.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(45,0), sticky="e")

        self.entry_rotV_1 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_1.insert(0,"0.0")
        self.entry_rotV_1.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_rotV_2 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_2.insert(0,"0.0")
        self.entry_rotV_2.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_rotV_3 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_3.insert(0,"0.0")
        self.entry_rotV_3.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_rotV = customtkinter.CTkButton(self.tabview.tab("Rotation vector"), text="Apply", command=self.apply_rotV, width=180)
        self.button_rotV.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Euler angles
        self.tabview.tab("Euler angles").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Euler angles").grid_columnconfigure(1, weight=0)
        
        self.label_EA_roll= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="roll:")
        self.label_EA_roll.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_EA_pitch= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="pitch:")
        self.label_EA_pitch.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_EA_yaw= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="yaw:")
        self.label_EA_yaw.grid(row=2, column=0, rowspan=3, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_EA_roll = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_roll.insert(0,"0.0")
        self.entry_EA_roll.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_EA_pitch = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_pitch.insert(0,"0.0")
        self.entry_EA_pitch.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_EA_yaw = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_yaw.insert(0,"0.0")
        self.entry_EA_yaw.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_EA = customtkinter.CTkButton(self.tabview.tab("Euler angles"), text="Apply", command=self.apply_EA, width=180)
        self.button_EA.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Quaternion
        self.tabview.tab("Quaternion").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Quaternion").grid_columnconfigure(1, weight=0)
        
        self.label_quat_0= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q0:")
        self.label_quat_0.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_quat_1= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q1:")
        self.label_quat_1.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_2= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q2:")
        self.label_quat_2.grid(row=2, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_3= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q3:")
        self.label_quat_3.grid(row=3, column=0, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_quat_0 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_0.insert(0,"1.0")
        self.entry_quat_0.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_quat_1 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_1.insert(0,"0.0")
        self.entry_quat_1.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_2 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_2.insert(0,"0.0")
        self.entry_quat_2.grid(row=2, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_3 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_3.insert(0,"0.0")
        self.entry_quat_3.grid(row=3, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_quat = customtkinter.CTkButton(self.tabview.tab("Quaternion"), text="Apply", command=self.apply_quat, width=180)
        self.button_quat.grid(row=4, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Rotation matrix info
        self.RotMFrame = customtkinter.CTkFrame(self, width=150)
        self.RotMFrame.grid(row=1, column=1, rowspan=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.RotMFrame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.label_RotM= customtkinter.CTkLabel(self.RotMFrame, text="RotM = ")
        self.label_RotM.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(20,0), sticky="e")

        self.entry_RotM_11= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_11.insert(0,"1.0")
        self.entry_RotM_11.configure(state="disabled")
        self.entry_RotM_11.grid(row=0, column=1, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_12= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_12.insert(0,"0.0")
        self.entry_RotM_12.configure(state="disabled")
        self.entry_RotM_12.grid(row=0, column=2, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_13= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_13.insert(0,"0.0")
        self.entry_RotM_13.configure(state="disabled")
        self.entry_RotM_13.grid(row=0, column=3, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_21= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_21.insert(0,"0.0")
        self.entry_RotM_21.configure(state="disabled")
        self.entry_RotM_21.grid(row=1, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_22= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_22.insert(0,"1.0")
        self.entry_RotM_22.configure(state="disabled")
        self.entry_RotM_22.grid(row=1, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_23= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_23.insert(0,"0.0")
        self.entry_RotM_23.configure(state="disabled")
        self.entry_RotM_23.grid(row=1, column=3, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_31= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_31.insert(0,"0.0")
        self.entry_RotM_31.configure(state="disabled")
        self.entry_RotM_31.grid(row=2, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_32= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_32.insert(0,"0.0")
        self.entry_RotM_32.configure(state="disabled")
        self.entry_RotM_32.grid(row=2, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_33= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_33.insert(0,"1.0")
        self.entry_RotM_33.configure(state="disabled")
        self.entry_RotM_33.grid(row=2, column=3, padx=(2,0), pady=(2,0), sticky="ew")
    
    def actulizarRot(self,R):

            self.entry_RotM_11.configure(state="normal")
            self.entry_RotM_11.delete(0,"end")
            self.entry_RotM_11.insert(0,str(R[0][0]))
            self.entry_RotM_11.configure(state="disabled")

            self.entry_RotM_12.configure(state="normal")
            self.entry_RotM_12.delete(0,"end")
            self.entry_RotM_12.insert(0,str(R[0][1]))
            self.entry_RotM_12.configure(state="disabled")

            self.entry_RotM_13.configure(state="normal")
            self.entry_RotM_13.delete(0,"end")
            self.entry_RotM_13.insert(0,str(R[0][2]))
            self.entry_RotM_13.configure(state="disabled")

            self.entry_RotM_21.configure(state="normal")
            self.entry_RotM_21.delete(0,"end")
            self.entry_RotM_21.insert(0,str(R[1][0]))
            self.entry_RotM_21.configure(state="disabled")

            self.entry_RotM_22.configure(state="normal")
            self.entry_RotM_22.delete(0,"end")
            self.entry_RotM_22.insert(0,str(R[1][1]))
            self.entry_RotM_22.configure(state="disabled")

            self.entry_RotM_23.configure(state="normal")
            self.entry_RotM_23.delete(0,"end")
            self.entry_RotM_23.insert(0,str(R[1][2]))
            self.entry_RotM_23.configure(state="disabled")

            self.entry_RotM_31.configure(state="normal")
            self.entry_RotM_31.delete(0,"end")
            self.entry_RotM_31.insert(0,str(R[2][0]))
            self.entry_RotM_31.configure(state="disabled")

            self.entry_RotM_32.configure(state="normal")
            self.entry_RotM_32.delete(0,"end")
            self.entry_RotM_32.insert(0,str(R[2][1]))
            self.entry_RotM_32.configure(state="disabled")

            self.entry_RotM_33.configure(state="normal")
            self.entry_RotM_33.delete(0,"end")
            self.entry_RotM_33.insert(0,str(R[2][2]))
            self.entry_RotM_33.configure(state="disabled")

            R0=np.array([[0.,0.,0],[0.,0.,0.],[0.,0.,0.]])
            
            Ri=np.array([[1.,0.,0],[0.,1.,0.],[0.,0.,1.]])

          

            
            axis,angle=rotMat2Eaa(R)
            if np.array_equal(R,R0):
                axis=np.array([1,0,0])
            if np.array_equal(R,Ri):
                axis=np.array([1,0,0])

               
            
            self.entry_AA_angle.delete(0,"end")
            self.entry_AA_angle.insert(0,str(angle*180/np.pi))
            self.entry_AA_ax1.delete(0,"end")
            self.entry_AA_ax1.insert(0,str(axis[0]))
            self.entry_AA_ax2.delete(0,"end")
            self.entry_AA_ax2.insert(0,str(axis[1]))
            self.entry_AA_ax3.delete(0,"end")
            self.entry_AA_ax3.insert(0,str(axis[2]))

            print("angle: ", angle)
            self.r=angle*axis
            print("r: ", self.r)

           

            yaw,pitch,roll=rotM2eAngles(R)

            self.entry_EA_yaw.delete(0,"end")
            self.entry_EA_yaw.insert(0,str(yaw))
            self.entry_EA_pitch.delete(0,"end")
            self.entry_EA_pitch.insert(0,str(pitch))
            self.entry_EA_roll.delete(0,"end")
            self.entry_EA_roll.insert(0,str(roll))

            self.entry_rotV_1.delete(0,"end")
            self.entry_rotV_1.insert(0,str(self.r[0]))
            self.entry_rotV_2.delete(0,"end")
            self.entry_rotV_2.insert(0,str(self.r[1]))
            self.entry_rotV_3.delete(0,"end")
            self.entry_rotV_3.insert(0,str(self.r[2]))
            
            qx=math.sin(angle)*axis
            Q=np.array([math.cos(angle/2),qx[0],qx[1],qx[2]])

            self.entry_quat_0.delete(0,"end")
            self.entry_quat_0.insert(0,str(Q[0]))
            self.entry_quat_1.delete(0,"end")
            self.entry_quat_1.insert(0,str(Q[1]))
            self.entry_quat_2.delete(0,"end")
            self.entry_quat_2.insert(0,str(Q[2]))
            self.entry_quat_3.delete(0,"end")
            self.entry_quat_3.insert(0,str(Q[3]))
            






    def resetbutton_pressed(self):
        """
        Event triggered function on the event of a push on the button Reset
        """  
        self.M = np.array(
            [[ -1,  -1, 1], #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.actulizarRot(R)
        self.update_cube() #Update the cube

        self.Q=np.array([1.,0.,0.,0.])
        # self.entry_AA_angle.delete(0,30)
        # self.entry_AA_angle.insert(0,str(0))
        # self.entry_AA_ax1.delete(0,30)
        # self.entry_AA_ax1.insert(0,str(1))
        # self.entry_AA_ax2.delete(0,30)
        # self.entry_AA_ax2.insert(0,str(0))
        # self.entry_AA_ax3.delete(0,30)
        # self.entry_AA_ax3.insert(0,str(0))


        # self.entry_rotV_1.delete(0,30)
        # self.entry_rotV_1.insert(0,str(0))
        # self.entry_rotV_2.delete(0,30)
        # self.entry_rotV_2.insert(0,str(0))
        # self.entry_rotV_3.delete(0,30)
        # self.entry_rotV_3.insert(0,str(0))
        pass

    
    def apply_AA(self):
        """
        Event triggered function on the event of a push on the button button_AA
        """
        #Example on hot to get values from entries:
        angle =float(self.entry_AA_angle.get())*np.pi/180
        axis1=float(self.entry_AA_ax1.get())
        axis2=float(self.entry_AA_ax2.get())
        axis3=float(self.entry_AA_ax3.get())
        axis= np.array([axis1,axis2,axis3])
        R=Eaa2rotM(angle,axis)
        if axis1==float(0) and axis2==float(0) and axis3== float(0):
         R= np.array([[1.,0.,0],[0.,1.,0.],[0.,0.,1.]])
        self.actulizarRot(R)
        self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
        self.update_cube() #Update the cube
        

    
    def apply_rotV(self):
        """
        Event triggered function on the event of a push on the button button_rotV 
        """
        rot_vec=np.array([0.,0.,0.])
        r0=np.array([0.,0.,0.])
        rot_vec[0]=float(self.entry_rotV_1.get())
        rot_vec[1]=float(self.entry_rotV_2.get())
        rot_vec[2]=float(self.entry_rotV_3.get())
        print("rotvec: ", rot_vec)
        R=rot_vecToRotM(rot_vec)
        if np.array_equal(rot_vec,r0):
           R= np.array([[1.,0.,0],[0.,1.,0.],[0.,0.,1.]])
        self.actulizarRot(R)
        self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
        self.update_cube() #Update the cube
        pass

    
    def apply_EA(self):
        """
        Event triggered function on the event of a push on the button button_EA
        """
        yaw=float(self.entry_EA_yaw.get())*np.pi/180
        pitch=float(self.entry_EA_pitch.get())*np.pi/180
        roll=float(self.entry_EA_roll.get())*np.pi/180
        R=eAngles2rotM(yaw,pitch,roll)
        self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
        self.actulizarRot(R)
        print(R)
        self.update_cube() #Update the cube
        pass

    
    def apply_quat(self):
        """
        Event triggered function on the event of a push on the button button_quat
        """
        q0=float(self.entry_quat_0.get())
        q1=float(self.entry_quat_1.get())
        q2=float(self.entry_quat_2.get())
        q3=float(self.entry_quat_3.get())
        Q=np.array([q0,q1,q2,q3])
        R = quaternion_rotation_matrix(Q)
        self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
        self.actulizarRot(R)
        print(R)
        self.update_cube() #Update the cube

        pass

    
    def onclick(self, event):
        """
        Event triggered function on the event of a mouse click inside the figure canvas
        """
        print("Pressed button", event.button)
  

        if event.button:
         x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates
         self.m0=creacvector_m(x_fig,y_fig)
         print("m0: ",self.m0)
         self.pressed = True # Bool to control(activate) a drag (click+move)
            
  
           

    def onmove(self,event):
        """
        Event triggered function on the event of a mouse motion
        """
        
          

        if self.pressed: #Only triggered if previous click
            x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates
            self.m1=creacvector_m(x_fig,y_fig)
            print("m1: ",self.m1)
            print("m0: ",self.m0)
            self.Q=fromtwovectors(self.m0,self.m1)
            print(self.Q)
            R=quaternion_rotation_matrix(self.Q)      
            self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
            self.actulizarRot(R)
            self.update_cube() #Update the cube
            self.m0=self.m1


    def onrelease(self,event):
        """
        Event triggered function on the event of a mouse release
        """
        self.pressed = False # Bool to control(deactivate) a drag (click+move)
        


    def init_cube(self):
        """
        Initialization function that sets up cube's geometry and plot information
        """

        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose() #Node 7

        self.con = [
            [0, 1, 2, 3], #Face 1
            [4, 5, 6, 7], #Face 2
            [3, 2, 6, 7], #Face 3
            [0, 1, 5, 4], #Face 4
            [0, 3, 7, 4], #Face 5
            [1, 2, 6, 5]] #Face 6

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]],self.M[:,row[3]]])

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')

        for item in [self.fig, ax]:
            item.patch.set_visible(False)

        self.facesObj = Poly3DCollection(faces, linewidths=.2, edgecolors='k',animated = True)
        self.facesObj.set_facecolor([(0,0,1,0.9), #Blue
        (0,1,0,0.9), #Green
        (.9,.5,0.13,0.9), #Orange
        (1,0,0,0.9), #Red
        (1,1,0,0.9), #Yellow
        (0,0,0,0.9)]) #Black

        #Transfering information to the plot
        ax.add_collection3d(self.facesObj)

        #Configuring the plot aspect
        ax.azim=-90
        ax.roll = -90
        ax.elev=0   
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)        
        ax.set_aspect('equal')
        ax.disable_mouse_rotation()
        ax.set_axis_off()

        self.pix2unit = 1.0/60 #ratio for drawing the cube 


    def update_cube(self):
        """
        Updates the cube vertices and updates the figure.
        Call this function after modifying the vertex matrix in self.M to redraw the cube
        """

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]], self.M[:,row[3]]])

        self.facesObj.set_verts(faces)
        self.bm.update()


    def canvas_coordinates_to_figure_coordinates(self,x_can,y_can):
        """
        Remap canvas coordinates to cube centered coordinates
        """

        (canvas_width,canvas_height)=self.canvas.get_width_height()
        figure_center_x = canvas_width/2+14
        figure_center_y = canvas_height/2+2
        x_fig = (x_can-figure_center_x)*self.pix2unit
        y_fig = (y_can-figure_center_y)*self.pix2unit

        return(x_fig,y_fig)


    def destroy(self):
        """
        Close function to properly destroy the window and tk with figure
        """
        try:
            self.destroy()
        finally:
            exit()


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
            cv.draw_idle()


if __name__ == "__main__":
    app = Arcball()
    app.mainloop()
    exit()
