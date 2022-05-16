# -*- coding: utf-8 -*-
#=========================================================
# Beginning of atomic_motion.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains everything needed for the 
# visualization of phonons in systems. It takes JSON 
# format #2 for performance
#
# This software is part of a package distributed under the 
# GPLv3 license, see ..LICENSE.txt
#=========================================================
import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate, ortho
from vispy.geometry import meshdata as md
from vispy.geometry import generation as gen
import imageio
from tqdm import trange
from .bonds import vesta_radius, vesta_colors, covalent_radii, atomic_number
from PyQt5 import QtWidgets

app.use_app(backend_name="PyQt5", call_reuse=True)

class Atom:

    def __init__(self, name, atomic_id, position):
        self.name = name 
        self.id = atomic_id
        self.xyz = position
        self.xyzt = self.xyz
        self.n_updates = 0
        self

    def updateXYZ(self, new_position):
        self.xyzt = np.vstack((self.xyzt, new_position))
        self.n_updates += 1

    def getXYZ(self):
        return self.xyz

    def getXYZT(self):
        return self.xyzt
    
    def getname(self):
        return self.name

    def getid(self):
        return self.id

    def __str__(self):
        text = ""
        text += "name: %s\n"%self.name
        text += "atom:\n"
        text += "%3s %3d"%(self.xyz,self.id) + "\n"
        return text

class Vibrations:

    def __init__(self, types, qpoints, atomic_positions, eigenvectors):
        self.types = types
        self.qpoints = qpoints
        self.atom_pos_red = atomic_positions
        self.eigenvectors = eigenvectors
        self.natoms = self.atom_pos_red.shape[0]

        self.dt = 0.001 #time step in seconds
        self.time = 0
        self.atoms = [Atom(self.types[i], i, self.atom_pos_red[i]) for i in range(self.natoms)]

    def getAtoms(self):
        return self.atoms

    def getVibration(self, index_q, index_nu):
        veckn = self.eigenvectors[index_q, index_nu, ...]
        qpt   = self.qpoints[index_q,...]
        self.atom_phase = np.zeros((3,1))
        vibrations = np.zeros((self.natoms, 3), dtype=complex)
        for i in range(self.natoms):
            self.atom_phase[i] = qpt.dot(self.atom_pos_red[i])

        for i in range(self.natoms):
            sprod = qpt.dot(np.array((1,1,1)))+ self.atom_phase[i] #change 111 depending on supercell
            arg = sprod*2.0*np.pi
            phase = complex(np.cos(arg), np.sin(arg))

            #displacement of the atoms
            x = complex(veckn[i,0,0], veckn[i,0,1])*phase
            y = complex(veckn[i,1,0], veckn[i,1,1])*phase
            z = complex(veckn[i,2,0], veckn[i,2,1])*phase
            vibrations[i] = np.array((x,y,z))
        return vibrations

DEFAULT_COLOR = (0, 0.2, 0.2, 0.5)

def get_combinations(elements):
    combos = list()
    for i in range(len(elements)):
        for j in range(len(elements)):
            combos.append([i,j])
    return combos

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.sqrt(dist_2), np.argmin(dist_2)

cart2polar = lambda x,y,z: (np.sqrt(x * x + y * y + z * z),np.arccos(x / np.sqrt(x * x + y * y)) * (-1 if y>0 else 1) * 180/np.pi ,np.arccos(z / np.sqrt(x * x + y * y + z * z)) * 180/np.pi)

class MyMeshData(md.MeshData):
    """ Add to Meshdata class the capability to export good data for gloo """

    def __init__(self, vertices=None, faces=None, edges=None,
                 vertex_colors=None, face_colors=None):
        md.MeshData.__init__(self, vertices=None, faces=None, edges=None,
                             vertex_colors=None, face_colors=None)

    def get_glTriangles(self):
        """
        Build vertices for a colored mesh.
            V  is the vertices
            I1 is the indices for a filled mesh (use with GL_TRIANGLES)
            I2 is the indices for an outline mesh (use with GL_LINES)
        """
        vtype = [('a_position', np.float32, 3),
                 ('a_normal', np.float32, 3),
                 ('a_color', np.float32, 4)]
        vertices = self.get_vertices()
        normals = self.get_vertex_normals()
        faces = np.uint32(self.get_faces())

        edges = np.uint32(self.get_edges().reshape((-1)))
        colors = self.get_vertex_colors()

        nbrVerts = vertices.shape[0]
        V = np.zeros(nbrVerts, dtype=vtype)
        V[:]['a_position'] = vertices
        V[:]['a_normal'] = normals
        V[:]['a_color'] = colors

        return V, faces.reshape((-1)), edges.reshape((-1))


vert = """
// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;

// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;

// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    v_color = a_color * u_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""


frag = """
// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""

vertex = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;

void main (void) {
    v_radius = a_radius;
    v_color = a_color;

    v_eye_position = u_view * u_model * vec4(a_position,1.0);
    v_light_direction = normalize(u_light_position);
    float dist = length(v_eye_position.xyz);

    gl_Position = u_projection * v_eye_position;

    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    vec4  proj_corner = u_projection * vec4(a_radius, a_radius, v_eye_position.z, v_eye_position.w);  // # noqa
    gl_PointSize = 512.0 * proj_corner.x / proj_corner.w;
}
"""

fragment = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
void main()
{
    // r^2 = (x - x0)^2 + (y - y0)^2 + (z - z0)^2
    vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    if (d <= 0.0)
        discard;

    float z = sqrt(d);
    vec4 pos = v_eye_position;
    pos.z += v_radius*z;
    vec3 pos2 = pos.xyz;
    pos = u_projection * pos;
//    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
    vec3 normal = vec3(x,y,z);
    float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);

    // Specular lighting.
    vec3 M = pos2.xyz;
    vec3 O = v_eye_position.xyz;
    vec3 L = u_light_spec_position;
    vec3 K = normalize(normalize(L - M) + normalize(O - M));
    // WARNING: abs() is necessary, otherwise weird bugs may appear with some
    // GPU drivers...
    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
    vec3 v_light = vec3(1., 1., 1.);
    gl_FragColor.rgba = vec4(.15*v_color + .55*diffuse * v_color
                        + .35*specular * v_light, 1.0);
}
"""
class VisualizerCanvas(app.Canvas):

    def __init__(self, input_json_data, show_bonds=False):
        app.Canvas.__init__(self, title='Molecular viewer',
                            keys='interactive', size=(1200, 800))
        self.input_json_data = input_json_data
        self.ps = self.pixel_scale
        self.size = 1200, 800
        self.translate = 40
        self.program = gloo.Program(vertex, fragment)        
        self.view = translate((0, 0, -self.translate))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.NX = 4
        self.NY = 4
        self.NZ = 1
        self.load_molecule()
        self.load_data()
        self.theta = 0
        self.phi = 0
        self.time = 0
        self.dt = 0.01
        self.phase = 0
        self.amplitude = 0.06
        self.speed = 2

        #init params for bonds
        # self.nBonds = 6
        self.draw_bonds_bool = show_bonds
        self.programs_bonds = [gloo.Program(vert, frag) for _ in range(self.nBonds)]
        self.meshes = [MyMeshData() for _ in range(self.nBonds)]
        self.vertices = list()
        self.filled = list()
        self.vertices_buf = list()
        self.filled_buf = list()
        self.outline_buf = list()
        #bond sets
        self.set_bond_program_param('u_model', self.model)
        self.set_bond_program_param('u_view', self.view)

        self.apply_zoom()

        self.stop_rotation = False
        gloo.set_state(depth_test=True, clear_color=(0.30, 0.30, 0.35, 1.00))
        self.timer = app.Timer(1.0 / 60, connect=self.on_timer, start=True)
        self.visible = True
        # self.show()

    def local_append(self, orig, appendee):
        return appendee if orig is None else np.append(orig, appendee, axis=0)

    def load_molecule(self, idq=2348, nu=1):

        ATOM_POS_CAR = np.array(self.input_json_data['atom_pos_car'])
        ATOM_POS_RED = np.array(self.input_json_data['atom_pos_red'])
        NATOMS = self.input_json_data['natoms']
        LAT = np.array(self.input_json_data['lattice'])
        ATOM_TYPES = self.input_json_data['atom_types']
        QPOINTS = np.asarray(self.input_json_data['qpoints'])

        EIGENVECTORS = np.asarray(self.input_json_data['vectors']) 

        self.atom_numbers = list()
        molecule = np.zeros((self.NX*self.NY*self.NZ*NATOMS, 7))
        idt = 0
        for nx in range(self.NX):
            for ny in range(self.NY):
                for nz in range(self.NZ):
                    NEW_ORIGIN = LAT[0,:] * nx + LAT[1,:] * ny + LAT[2,:] * nz
                    for i in range(NATOMS):
                        molecule[idt,:3] = NEW_ORIGIN + ATOM_POS_CAR[i,:]
                        molecule[idt, 3:6] = np.array(vesta_colors[self.input_json_data['atom_numbers'][i]])
                        molecule[idt, 6] = vesta_radius[atomic_number[self.input_json_data['atom_types'][i]]]# 2 if i==0 else 0.75
                        self.atom_numbers.append(self.input_json_data['atom_numbers'][i])
                        idt += 1
        self._nAtoms = molecule.shape[0]

        # The x,y,z values store in one array
        self.coords = molecule[:, :3]
        self.coords[:,0] -= np.average(self.coords[:,0])
        self.coords[:,1] -= np.average(self.coords[:,1])

        # The array that will store the color and alpha scale for all the atoms
        self.atomsColours = molecule[:, 3:6]
        # The array that will store the scale for all the atoms.
        self.atomsScales = molecule[:, 6]

        #get displacement data
        self.idq = idq
        self.nu = nu
        self.motions = Vibrations(ATOM_TYPES, QPOINTS, ATOM_POS_RED, EIGENVECTORS)
        self.vibrations = self.motions.getVibration(self.idq, self.nu)

    def load_data(self):
        n = self._nAtoms
        #get nearest neighbor distance
        combinations = get_combinations(range(self._nAtoms))
        distances = np.sqrt(np.sum((self.coords[combinations[:][0],:]-self.coords[combinations[:][1],:])**2))
        self.nndist = distances[distances>0].min() + 0.05

        #determine bonds, across lattice
        self.bonds = list()
        for i in range(len(combinations)):
            a = combinations[i][0]
            b = combinations[i][1]
            ad = self.coords[a,:]
            bd = self.coords[b,:]
            length = np.sqrt(np.sum((ad-bd)**2))
            cra = covalent_radii[ self.atom_numbers[ a ]]
            crb = covalent_radii[ self.atom_numbers[ b ]]
            if (length < cra + crb) or (length < self.nndist):
                self.bonds.append([a,b])
        self.nBonds = len(self.bonds)
        if self.nBonds > 40:
            print("HEADS UP this is a lot of bonds, and is gonna make this slow....")
            
        data = np.zeros(n, [('a_position', np.float32, 3),
                            ('a_color', np.float32, 3),
                            ('a_radius', np.float32)])

        data['a_position'] = self.coords
        data['a_color'] = self.atomsColours
        data['a_radius'] = self.atomsScales*self.ps


        self.program.bind(gloo.VertexBuffer(data))

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_light_position'] = 0., 0., 2.
        self.program['u_light_spec_position'] = -5., 5., -5.


    def rebind_data(self):
        data = np.zeros(self._nAtoms, [('a_position', np.float32, 3),
                            ('a_color', np.float32, 3),
                            ('a_radius', np.float32)])

        data['a_position'] = self.coords
        data['a_color'] = self.atomsColours
        data['a_radius'] = self.atomsScales*self.ps
        self.program.bind(gloo.VertexBuffer(data))

    def on_mouse_press(self, event):
        if event.button == 2:
            print("YES")

    def on_key_press(self, event):

        if event.text == ' ':
            self.stop_rotation = not self.stop_rotation
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

        if event.key in ['Left', 'Right']:
            if event.key == 'Right':
                self.theta += .5
            else:
                self.theta -= 0.5

        if event.key in ['Up', 'Down']:
            if event.key == 'Up':
                self.phi += .5
            else:
                self.phi -= 0.5

        if event.text in ['b','B']:
            self.draw_bonds_bool = not self.draw_bonds_bool

        # if event.text in ['c','C']:
        #     self.save_animation()

        if event.text in ['x','y','z', 'X', 'Y', 'Z']:
            if event.text.upper() == 'X':
                self.theta = 90
                self.phi = 90
            elif event.text.upper() == 'Y':
                self.theta = 0
                self.phi = 90
            else:
                self.theta = 0
                self.phi = 0

        if event.text in ['w', 's', 'a', 'd', 'W', 'S', 'A', 'D']:
            if event.text.upper() == 'W':
                self.coords[:,1] += 0.1
            elif event.text.upper() == 'S':
                self.coords[:,1] -= 0.1
            elif event.text.upper() == 'A':
                self.coords[:,0] -= 0.1
            elif event.text.upper() == 'D':
                self.coords[:,0] += 0.1
            
        if event.text in ['p','n','P','N']:
            width, height = self.physical_size
            if event.text.upper() == 'P':
                self.projection = perspective(25.0, width / float(height), 2.0, 100.0)
            else:
                self.projection = ortho(-10, 10, -10, 10, 2.0, 100.0)

        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))

        self.program['u_model'] = self.model
        self.program['u_projection'] = self.projection
        self.set_bond_program_param('u_projection', self.projection)
        self.set_bond_program_param('u_model', self.model)
        if self.draw_bonds_bool:
            self.draw_bonds()

        self.rebind_data()
        self.update()

    def on_timer(self, event):
        if not self.stop_rotation:
            arg = self.time * self.speed * 2.0 * np.pi
            phase = complex(self.amplitude*np.cos(arg), self.amplitude*np.sin(arg))
            self.phase += phase
            for idt in range(self._nAtoms):
                vx = np.real( phase * self.vibrations[idt % 3, 0])
                vy = np.real( phase * self.vibrations[idt % 3, 1])
                vz = np.real( phase * self.vibrations[idt % 3, 2])
                self.coords[idt,0] += vx
                self.coords[idt,1] += vy
                self.coords[idt,2] += vz

            self.rebind_data()
            self.time += self.dt

        self.program['u_model'] = self.model
        self.set_bond_program_param('u_model', self.model)
        if self.draw_bonds_bool:
            self.draw_bonds()
        self.update()

    def set_bond_program_param(self, param, val):
        for idc in range(self.nBonds):
            self.programs_bonds[idc][param] = val

    def return_time(self, only_time=False):
        if not only_time:
            self.timer.start()
            self.on_timer(event=None)
            self.timer.stop()
        return self.time, self.dt
        
    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(25.0, width / float(height), 2.0, 100.0)
        self.program['u_projection'] = self.projection
        self.set_bond_program_param('u_projection', self.projection)

    def apply_zoom(self):
        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(25.0, width / float(height), 2.0, 100.0)
        self.program['u_projection'] = self.projection
        self.set_bond_program_param('u_projection', self.projection)

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.translate = max(-1, self.translate)
        self.view = translate((0, 0, -self.translate))

        self.program['u_view'] = self.view
        self.set_bond_program_param('u_view', self.view)

        self.update()

    def draw_bonds(self):
        self.vertices = list()
        self.filled = list()
        self.vertices_buf = list()
        self.filled_buf = list()
        self.outline_buf = list()

        for idc, combo in enumerate(self.bonds[:self.nBonds]):
            a, b = combo
            x = self.coords[b,0] - self.coords[a,0]
            y = self.coords[b,1] - self.coords[a,1]
            z = self.coords[b,2] - self.coords[a,2]
            r, long, lat = cart2polar(x,y,z)

            local_bond_mesh = gen.create_cylinder(rows=1, cols=6,radius=[0.1,0.1], length=r)
            self.meshes[idc].set_vertices(local_bond_mesh.get_vertices())
            self.meshes[idc].set_faces(local_bond_mesh.get_faces())
            colors = np.tile(DEFAULT_COLOR, (self.meshes[idc].n_vertices, 1))
            self.meshes[idc].set_vertex_colors(colors)
            vertices, filled, outline = self.meshes[idc].get_glTriangles() #this bond is in local custom mesh class, at origin

            align = np.dot(rotate(long, (0, 0, 1)),
                           rotate(lat, (0, 1, 0)))
            for i in range(vertices.shape[0]):
                vertices[i]['a_position'] = -align[:3,:3]@(vertices[i]['a_position']) + self.coords[a,:]

            self.vertices.append(vertices)
            self.filled.append(filled)

            self.vertices_buf.append(gloo.VertexBuffer(vertices))
            self.filled_buf.append(gloo.IndexBuffer(filled))
            self.outline_buf.append(gloo.IndexBuffer(outline))

            self.programs_bonds[idc].bind(self.vertices_buf[idc])

    def on_draw(self, event):
        gloo.clear()
        if self.draw_bonds_bool:
            self.draw_bonds()
            # gloo.set_state(blend=False, depth_test=True,
                            #    polygon_offset_fill=True, clear_color='black')
            for idc in range(self.nBonds):
                self.programs_bonds[idc]['u_color'] = 1, 1, 1, 1
                self.programs_bonds[idc].draw('triangles', self.filled_buf[idc])

        # gloo.set_state(depth_test=True, clear_color='black')
        if self.visible:
            self.program.draw('points')

    def set_view(self, amplitude, speed):
        self.stop_rotation = True
        self.amplitude = amplitude
        self.speed = speed
        self.load_molecule()
        self.rebind_data()
        self.update()
        self.stop_rotation = not self.stop_rotation

    def set_motion(self, idq, nu):
        self.stop_rotation = True
        self.load_molecule(idq, nu) #reloads supercell in eq.
        self.rebind_data()
        self.update()
        self.stop_rotation = not self.stop_rotation

    def save_animation(self, fname="animation.gif"):
        try:
            writer = imageio.get_writer(fname, format='GIF-PIL', duration = self.dt)
            NSTEPS = int(1 / self.dt)
            for _ in trange(NSTEPS, desc="Saving animation"):
                im = self.render(alpha=True)
                time,_ = self.return_time()
                writer.append_data(im)
            writer.close()
            self.timer.start()
        except:
            print("Failed to save gif....")


class VisualizerWidget(QtWidgets.QMainWindow):
    def __init__(self, input_json_data):
        super().__init__()
        WIDTH = 800
        HEIGHT = 800
        self.resize(WIDTH, HEIGHT)
        self.canvas = VisualizerCanvas(input_json_data=input_json_data, show_bonds=False)
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self._main.setLayout(QtWidgets.QVBoxLayout())
        self._main.layout().addWidget(self.canvas.native)
    def keyPressEvent(self, event):
        print(event.text())
#=========================================================
# End of atomic_motion.py
#=========================================================