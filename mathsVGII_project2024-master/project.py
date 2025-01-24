import customtkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math
import numpy as np

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


def return_Axis_Amgle(R):
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


def Euler_Angles_to_Rot(yaw,pitch,roll): #psi, theta, phi
    """
    Converts Euler angles to a rotation matrix.
    Parameters:
        yaw (float): Rotation angle around the Z-axis (in radians).
        pitch (float): Rotation angle around the Y-axis (in radians).
        roll (float): Rotation angle around the X-axis (in radians).
    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # Rotation around the X-axis
    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # Rotation around the Y-axis
    rot_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # Rotation around the Z-axis
    rot_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine the rotations in the order Z -> Y -> X
    rotation_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)

    return rotation_matrix

def rotMa_To_Yaw_Pitch_Roll(matrix): #psi, theta, phi
    """
    Extracts Euler angles (yaw, pitch, roll) from a given rotation matrix.

    Parameters:
        matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: Yaw, pitch, and roll angles (in radians).
    """
    # Handle edge cases for gimbal lock
    if matrix[2, 0] == 1:
        pitch = math.pi / 2
        yaw = math.atan2(matrix[0, 1], matrix[0, 2])
        roll = 0
    elif matrix[2, 0] == -1:
        pitch = -math.pi / 2
        yaw = math.atan2(-matrix[0, 1], -matrix[0, 2])
        roll = 0
    else:
        pitch = math.asin(matrix[2, 0])
        roll = math.atan2(-matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(-matrix[1, 0], matrix[0, 0])

    return yaw, pitch, roll


def vector_to_RotMa(vector):    
    # Obtener la magnitud del vector y el eje de rotación
    norm = Magnitude(vector)
    unit_axis = vector / norm

    theta = norm  # Ángulo de rotación

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos = 1 - cos_theta

    u = unit_axis[0]  # Componente x
    v = unit_axis[1]  # Componente y
    w = unit_axis[2]  # Componente z

    uu = u * u
    vv = v * v
    ww = w * w

    # Crear la matriz de rotación
    rotation_matrix = np.array([
        [one_minus_cos * uu + cos_theta, one_minus_cos * u * v - sin_theta * w, one_minus_cos * u * w + sin_theta * v],
        [one_minus_cos * u * v + sin_theta * w, one_minus_cos * vv + cos_theta, one_minus_cos * v * w - sin_theta * u],
        [one_minus_cos * u * w - sin_theta * v, one_minus_cos * v * w + sin_theta * u, one_minus_cos * ww + cos_theta]
    ])

    print("Rotation Matrix:\n", rotation_matrix)
    return rotation_matrix

def Create_RotMa(theta, axis_vector):
    '''
    Computes the rotation matrix that rotates vectors by an angle 'theta' (in radians) about the specified axis 'axis_vector'.
    '''
    # Normalize the axis vector
    axis_length = Magnitude(axis_vector)
    normalized_axis = axis_vector / axis_length

    # Precompute trigonometric values
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos = 1 - cos_theta

    # Extract normalized axis components
    u = normalized_axis[0]
    v = normalized_axis[1]
    w = normalized_axis[2]

    # Compute the rotation matrix using the formula
    rotation_matrix = np.array([
        [one_minus_cos * u**2 + cos_theta, one_minus_cos * u * v - sin_theta * w, one_minus_cos * u * w + sin_theta * v],
        [one_minus_cos * u * v + sin_theta * w, one_minus_cos * v**2 + cos_theta, one_minus_cos * v * w - sin_theta * u],
        [one_minus_cos * u * w - sin_theta * v, one_minus_cos * v * w + sin_theta * u, one_minus_cos * w**2 + cos_theta]
    ])

    return rotation_matrix

def Quat_RotMa(Q):
    '''
    Converts a quaternion Q into a 3x3 rotation matrix.
    '''
    # Normalize the quaternion
    norm = math.sqrt(sum(component**2 for component in Q))
    Q = [q / norm for q in Q]

    # Extract components of the quaternion
    w, x, y, z = Q

    # Compute the elements of the rotation matrix
    m00 = w**2 + x**2 - y**2 - z**2
    m01 = 2 * (x * y - w * z)
    m02 = 2 * (x * z + w * y)

    m10 = 2 * (x * y + w * z)
    m11 = w**2 - x**2 + y**2 - z**2
    m12 = 2 * (y * z - w * x)

    m20 = 2 * (x * z - w * y)
    m21 = 2 * (y * z + w * x)
    m22 = w**2 - x**2 - y**2 + z**2

    # Assemble the rotation matrix
    rotation_matrix = np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])

    return rotation_matrix

def Two_Vectors_To_One(m0, m1):
    
    angle=math.acos(np.dot(m1,np.transpose(m0))/(Magnitude(m0)*Magnitude(m1)))
    print("angle: ",angle)
    q0=np.cos(angle/2)
    qx=np.sin(angle/2)*(np.cross(np.transpose(m0),np.transpose(m1))/(Magnitude(np.cross(m0,m1))))
    print("qx: ",qx)
    q1=qx[0]
    q2=qx[1]
    q3=qx[2]
    q=np.array([q0,q1,q2,q3])

    return q

def Magnitude(vector): 
 return np.sqrt(sum(pow(element, 2) for element in vector))

def Create_A_Vector(x,y):
    x2=x*x
    print("x2: ",x2)
    y2=y*y
    print("y2: ",y2)
    r2 = 4
    vex=np.array([y,r2/(2*np.sqrt(x2+y2)),-x])
    vexmodule= Magnitude(vex)
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
    
    def Rotation_Actualitation(self,R):
        '''
        Updates the GUI elements with the provided rotation matrix R and derived rotation data.
        '''
        # Update entries for the rotation matrix
        for i in range(3):
            for j in range(3):
                entry = getattr(self, f"entry_RotM_{i+1}{j+1}")
                entry.configure(state="normal")
                entry.delete(0, "end")
                entry.insert(0, str(R[i][j]))
                entry.configure(state="disabled")

        # Define zero and identity matrices
        zero_matrix = np.zeros((3, 3))
        identity_matrix = np.eye(3)

        # Compute axis-angle representation
        axis, angle = return_Axis_Amgle(R)
        if np.array_equal(R, zero_matrix) or np.array_equal(R, identity_matrix):
            axis = np.array([1, 0, 0])

        # Update axis-angle GUI elements
        self.entry_AA_angle.delete(0, "end")
        self.entry_AA_angle.insert(0, str(angle * 180 / np.pi))
        for i, axis_value in enumerate(axis):
            entry = getattr(self, f"entry_AA_ax{i+1}")
            entry.delete(0, "end")
            entry.insert(0, str(axis_value))

        # Compute rotation vector
        self.r = angle * axis
        for i, r_value in enumerate(self.r):
            entry = getattr(self, f"entry_rotV_{i+1}")
            entry.delete(0, "end")
            entry.insert(0, str(r_value))

        # Compute yaw, pitch, roll and update entries
        yaw, pitch, roll = rotMa_To_Yaw_Pitch_Roll(R)
        self.entry_EA_yaw.delete(0, "end")
        self.entry_EA_yaw.insert(0, str(yaw))
        self.entry_EA_pitch.delete(0, "end")
        self.entry_EA_pitch.insert(0, str(pitch))
        self.entry_EA_roll.delete(0, "end")
        self.entry_EA_roll.insert(0, str(roll))

        # Compute quaternion and update entries
        qx = math.sin(angle / 2) * axis
        Q = np.array([math.cos(angle / 2), qx[0], qx[1], qx[2]])
        for i, q_value in enumerate(Q):
            entry = getattr(self, f"entry_quat_{i}")
            entry.delete(0, "end")
            entry.insert(0, str(q_value))

        # Print debug information
        print("Angle:", angle)
        print("Rotation Vector:", self.r)

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
        self.Rotation_Actualitation(R)
        self.update_cube() #Update the cube
        pass

    
    def apply_AA(self):
        """
        Event handler for when the button button_AA is pressed.
        Applies a rotation to the object based on the Axis-Angle parameters entered.
        """
        # Retrieve angle and axis components from entries
        angle = float(self.entry_AA_angle.get()) * np.pi / 180  # Convert to radians
        axis = np.array([
            float(self.entry_AA_ax1.get()),
            float(self.entry_AA_ax2.get()),
            float(self.entry_AA_ax3.get())
        ])

        # Compute the rotation matrix
        if np.allclose(axis, 0):
            R = np.eye(3)  # Identity matrix if the axis is zero
        else:
            R = Create_RotMa(angle, axis)

        # Update rotation in the GUI and modify the object's transformation matrix
        self.Rotation_Actualitation(R)
        self.M = R @ self.M  # Update the transformation matrix
        
        # Update the visual representation of the object
        self.update_cube()
        


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
        R=vector_to_RotMa(rot_vec)
        if np.array_equal(rot_vec,r0):
           R= np.array([[1.,0.,0],[0.,1.,0.],[0.,0.,1.]])
        self.Rotation_Actualitation(R)
        self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
        self.update_cube() #Update the cube
        pass
        pass

    
    def apply_EA(self):
        """
        Event handler for when button_EA is pressed.
        Applies a rotation to the object using Euler angles (yaw, pitch, roll) from user inputs.
        """
        # Retrieve Euler angles from input fields and convert to radians
        yaw_angle = float(self.entry_EA_yaw.get()) * np.pi / 180
        pitch_angle = float(self.entry_EA_pitch.get()) * np.pi / 180
        roll_angle = float(self.entry_EA_roll.get()) * np.pi / 180

        # Compute the rotation matrix from Euler angles
        rotation_matrix = Euler_Angles_to_Rot(yaw_angle, pitch_angle, roll_angle)

        # Update the object's transformation matrix
        self.M = rotation_matrix @ self.M

        # Update the GUI with the new rotation matrix
        self.Rotation_Actualitation(rotation_matrix)

        # Print the resulting matrix for debugging
        print(rotation_matrix)

        # Refresh the cube visualization
        self.update_cube()
        pass

    
    def apply_quat(self):
        """
        Event handler for when button_quat is pressed.
        Applies a rotation to the object using quaternion values from user inputs.
        """
        # Retrieve quaternion components from input fields
        quat_w = float(self.entry_quat_0.get())
        quat_x = float(self.entry_quat_1.get())
        quat_y = float(self.entry_quat_2.get())
        quat_z = float(self.entry_quat_3.get())

        # Form the quaternion as a numpy array
        quaternion = np.array([quat_w, quat_x, quat_y, quat_z])

        # Compute the rotation matrix from the quaternion
        rotation_matrix = Quat_RotMa(quaternion)

        # Update the object's transformation matrix
        self.M = rotation_matrix @ self.M

        # Update the GUI with the new rotation matrix
        self.Rotation_Actualitation(rotation_matrix)

        # Print the resulting matrix for debugging
        print(rotation_matrix)

        # Refresh the cube visualization
        self.update_cube()

        pass

    
    def onclick(self, event):
        """
        Event triggered function on the event of a mouse click inside the figure canvas
        """
        print("Pressed button", event.button)
  

        if event.button:
         x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates
         self.m0=Create_A_Vector(x_fig,y_fig)
         print("m0: ",self.m0)
         self.pressed = True # Bool to control(activate) a drag (click+move)
            
    def onmove(self,event):
        """
        Event triggered function on the event of a mouse motion
        """
        
          

        if self.pressed: #Only triggered if previous click
            x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates
            self.m1=Create_A_Vector(x_fig,y_fig)
            print("m1: ",self.m1)
            print("m0: ",self.m0)
            self.Q=Two_Vectors_To_One(self.m0,self.m1)
            print(self.Q)
            R=Quat_RotMa(self.Q)      
            self.M = R.dot(self.M) #Modify the vertices matrix with a rotation matrix M
            self.Rotation_Actualitation(R)
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
