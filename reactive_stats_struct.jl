#information of grid, boundary condtions, landscape function, diffusion matrix 
Base.@kwdef mutable struct grid_info
    grid_stat :: Union{Grid{2, Triangle, Float64}, Missing} = missing
    grid_comm :: Union{Grid{2, Triangle, Float64}, Missing} = missing
    D::Union{Matrix{Function}, Missing} = missing

    g::Union{Vector{Function}, Missing} = missing

    dh_comm::Union{DofHandler{2, Grid{2, Triangle, Float64}},Missing}= missing #create_dofhandler(grid_comm);
    dh_stat::Union{DofHandler{2, Grid{2, Triangle, Float64}},Missing}= missing #create_dofhandler(grid_stat);

    ip = Lagrange{2, RefTetrahedron, 1}();

    cellvalues_forward = create_values(ip);
    cellvalues_backward = create_values(ip);
    cellvalues_stat = create_values(ip);

    dbc_for::Union{ConstraintHandler{DofHandler{2, Grid{2, Triangle, Float64}}, Float64}, Missing} = missing;
    dbc_back::Union{ConstraintHandler{DofHandler{2, Grid{2, Triangle, Float64}}, Float64}, Missing} = missing;
    dbc_stats::Union{ConstraintHandler{DofHandler{2, Grid{2, Triangle, Float64}}, Float64}, Missing} = missing;

end

#add boundary condtions and diffusion matrix 
function implement_grid(b::grid_info, D::Matrix{Function}, g::Vector{Function})
    ∂outer = obtain_boundary(b.grid_stat, 1);
    b.dbc_stats = create_bc(b.dh_stat,"stationary", b.cellvalues_stat, ∂outer);
    b.D = D
    b.g = g
end



#reactive statistics
Base.@kwdef mutable struct reactive_stat
    u_forward:: Union{Vector{Float64}, Missing} = missing
    u_stat:: Union{Vector{Float64}, Missing} = missing
    u_backward:: Union{Vector{Float64}, Missing} = missing
    grad_μ_projector::Union{L2Projector, Missing} = missing
    grad_μ_projected::Union{Vector{Vec{2, Float64}}, Missing} = missing
    Z_normalization::Float64 = 0
    transition_rate::Float64 =0
    reactive_density::Union{Vector{Vec{1, Float64}}, Missing} = missing
    reactive_current::Union{Vector{Vec{2, Float64}}, Missing} = missing
end


#implement grid for forward and backward committor function 
function implement_grid(range_loc::Vector{Vector{Float64}}, grid_loc::Vector{Vector{Float64}}, b::grid_info, react_stat::reactive_stat)
    #center and radius of attracting states 
    st_A = range_loc[1];ed_A = range_loc[2]; st_B = range_loc[3];ed_B = range_loc[4];
    A_center, A_radius = findcenter_radius(st_A, ed_A, b, react_stat)
    B_center, B_radius = findcenter_radius(st_B, ed_B, b, react_stat)
    A_radius = 0.25;
    B_radius = 0.25;
    b.grid_comm = create_rect_mesh_comm(A_center, A_radius, B_center, B_radius, grid_loc[1],grid_loc[2], grid_loc[3], grid_loc[4])
    b.dh_comm = create_dofhandler(b.grid_comm);
    ∂A, ∂B  = obtain_boundary(b.grid_comm, 0);
    ∂outer = obtain_boundary(b.grid_stat, 1);
    b.dbc_for = create_bc(b.dh_comm,"forward", b.cellvalues_forward, ∂A, ∂B, ∂outer);
    b.dbc_back = create_bc(b.dh_comm,"backward", b.cellvalues_backward, ∂A, ∂B, ∂outer);
end

function create_tri_mesh_comm(A_center, A_radius, B_center, B_radius, left_bott, left_upper, right_bott)
    gmsh.initialize()
    dim = Int64(gmsh.model.getDimension())
    a = -1
    b = left_upper[2] - a*left_bott[1]
    gmsh.model.occ.addPoint(left_bott[1], left_bott[2], 0, 0, 1)
    gmsh.model.occ.addPoint(left_upper[1], left_upper[2], 0, 0, 2)
    gmsh.model.occ.addPoint(right_bott[1], right_bott[2], 0, 0, 3)

    gmsh.model.occ.addLine(1, 2, 4)
    gmsh.model.occ.addLine(2, 3, 5)
    gmsh.model.occ.addLine(3, 1, 6)

    gmsh.model.occ.addCurveLoop([4, 5, 6], 1)

    tri = gmsh.model.occ.addPlaneSurface([1], 1)

    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [tri], 2)
    #######

    #hole1 = gmsh.model.occ.addDisk(A_center[1], A_center[2], 0, A_radius+0.15, A_radius)
    #hole2 = gmsh.model.occ.addDisk(B_center[1], B_center[2], 0, B_radius+0.15, B_radius)

    hole1 = gmsh.model.occ.addDisk(A_center[1], A_center[2], 0, A_radius, A_radius)
    hole2 = gmsh.model.occ.addDisk(B_center[1], B_center[2], 0, B_radius, B_radius)

   #gmsh.model.occ.rotate([(2, hole1)], A_center[1], A_center[2], 0, 0, 0, 1, pi/2)

    membrane1 = gmsh.model.occ.cut([(2, 1)], [(2, hole1), (2,hole2)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities()
    gdim = 2
    gmsh.model.addPhysicalGroup(volumes[1][1], [volumes[1][2]], 11)

    boundary = gmsh.model.getBoundary(membrane1[1])
    boundary_ids = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(2, boundary_ids)

    #meshSize = 0.04
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.035)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.035)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.reverse()
    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1 

    nodes = tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    # "Domain" is the name of a PhysicalGroup and saves all cells that define the computational domain
    domaincellset = cellsets["12"]
    elements = elements[collect(domaincellset)]


    boundarydict = toboundary(facedim)
    facesets = tofacesets(boundarydict, elements)
    gmsh.finalize()
    grid = Grid(elements, nodes, facesets=facesets, cellsets=cellsets)

    addfaceset!(grid, "top", x -> x[2] ≈ a*x[1]+b);
    addfaceset!(grid, "left", x -> x[1] ≈ left_upper[1]);
    addfaceset!(grid, "right", x -> x[1] ≈ 20);
    addfaceset!(grid, "bottom", x -> x[2] ≈ right_bott[2]);
    #addfaceset!(grid, "A", x -> (x[1]-A_center[1])^2/(A_radius)^2+(x[2]-A_center[2])^2/(A_radius+0.15)^2 ≈ 1);
    #addfaceset!(grid, "B", x -> (x[1]-B_center[1])^2/(B_radius+0.15)^2+(x[2]-B_center[2])^2/(B_radius)^2 ≈ 1);
    addfaceset!(grid, "A", x -> (x[1]-A_center[1])^2+(x[2]-A_center[2])^2 ≈ A_radius^2);
    addfaceset!(grid, "B", x -> (x[1]-B_center[1])^2+(x[2]-B_center[2])^2 ≈ B_radius^2);
    return grid        
end

#implement grid for forward and backward committor function 
function implement_grid_tri(range_loc::Vector{Vector{Float64}}, grid_loc::Vector{Vector{Float64}}, b::grid_info, react_stat::reactive_stat)
    #center and radius of attracting states 
    st_A = range_loc[1];ed_A = range_loc[2]; st_B = range_loc[3];ed_B = range_loc[4];
    A_center, A_radius = findcenter_radius(st_A, ed_A, b, react_stat)
    B_center, B_radius = findcenter_radius(st_B, ed_B, b, react_stat)
    A_radius = 0.35;
    B_radius = 0.35;
    b.grid_comm = create_tri_mesh_comm(A_center, A_radius, B_center, B_radius, grid_loc[1],grid_loc[2], grid_loc[3])
    b.dh_comm = create_dofhandler(b.grid_comm);
    ∂A, ∂B  = obtain_boundary(b.grid_comm, 0);
    ∂outer = obtain_boundary(b.grid_stat, 1);
    b.dbc_for = create_bc(b.dh_comm,"forward", b.cellvalues_forward, ∂A, ∂B, ∂outer);
    b.dbc_back = create_bc(b.dh_comm,"backward", b.cellvalues_backward, ∂A, ∂B, ∂outer);
end

#implement grid for forward and backward committor function with boundary obtained by spline 
function implement_grid_spl(range_loc::Vector{Vector{Float64}}, b::grid_info, react_stat::reactive_stat)
    #center and radius of attracting states
    b.grid_comm = create_rect_mesh_comm_spl(b, react_stat, range_loc)
    b.dh_comm = create_dofhandler(b.grid_comm);
    ∂A, ∂B  = obtain_boundary(b.grid_comm, 0);
    ∂outer = obtain_boundary(b.grid_stat, 1);
    b.dbc_for = create_bc(b.dh_comm,"forward", b.cellvalues_forward, ∂A, ∂B, ∂outer);
    b.dbc_back = create_bc(b.dh_comm,"backward", b.cellvalues_backward, ∂A, ∂B, ∂outer);
end


function export_result(base_info::grid_info, react_stat::reactive_stat, 
    stat_loc::Vector{Vector{Float64}}, current_loc::Vector{Vector{Float64}}, level::Int64)
    mfpt = react_stat.Z_normalization /react_stat.transition_rate
    _,_ ,_ ,_ , current_points = trans_stat(current_loc, base_info, react_stat, level);
    _,_ ,u_points_stat , distribution_points, _ = trans_stat(stat_loc, base_info, react_stat, 200);
    return react_stat.transition_rate, mfpt, u_points_stat, distribution_points, current_points
end

function arrow0!(x, y, u, v; as=0.1, lw=1, lc=:black, la=1)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = as*nuv*v4, as*nuv*v5
    plot!([x,x+u], [y,y+v], lw=lw, lc=lc, la=la, label = false)
    plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], lw=lw, lc=lc, la=la, label = false)
    plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], lw=lw, lc=lc, la=la, label = false)
end
