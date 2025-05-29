
# create interpolation for cellvalues
function create_values(interpolation_q_forward)
    # quadrature rules
    dim = 2
    # geometric interpolation
    #ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(2)
    # cellvalues for q_forward
    cellvalues_q_forward = CellScalarValues(qr, interpolation_q_forward)
    #cellvalues for q_backward
    return cellvalues_q_forward
end;

#obtain the face boundary for grids
function obtain_boundary(grid, check::Int)
    if check == 0
        ∂A = union(
            getfaceset(grid, "A"),
        );

        ∂B = union(
            getfaceset(grid, "B"),
        );
        return ∂A, ∂B
    else
        ∂outer = union(
            getfaceset(grid, "top"),
            getfaceset(grid, "bottom"),
            getfaceset(grid, "left"),
            getfaceset(grid, "right"),
        );
        return ∂outer
    end
end

#create degree of freeddm handler for variable 
function create_dofhandler(grid)
    dh = DofHandler(grid)
    add!(dh, :u, 1) 
    close!(dh)
    return dh
end;

#build up the boundry condtions
function create_bc(dh, type::String, cellvalues::CellScalarValues, ∂A::Set{FaceIndex}, ∂B::Set{FaceIndex}, ∂outer::Set{FaceIndex}) # 0 for forward, 1 for backward, 2 for stationary 
    dbc = ConstraintHandler(dh)
    if type == "forward"
        dbc_inner1 = Dirichlet(:u, ∂A, (x,t) -> 0)
        dbc_inner2 = Dirichlet(:u, ∂B, (x,t) -> 1)
    elseif type == "backward"
        dbc_inner1 = Dirichlet(:u, ∂A, (x,t) -> 1)
        dbc_inner2 = Dirichlet(:u, ∂B, (x,t) -> 0)
    end
    if type != "stationary" 
        add!(dbc, dbc_inner1);
        add!(dbc, dbc_inner2);
    else
        dbc_outer = Dirichlet(:u, ∂outer, (x,t) -> 0)
        mean_value_constraint = setup_mean_constraint(cellvalues, dh)
        add!(dbc, dbc_outer);
        add!(dbc, mean_value_constraint);
    end
    close!(dbc)
    return dbc
end;

#build up the boundry condtions for stationary distribution 
function create_bc(dh, type::String, cellvalues::CellScalarValues, ∂outer::Set{FaceIndex}) # 0 for forward, 1 for backward, 2 for stationary 
    dbc = ConstraintHandler(dh)
    if type == "stationary" 
        dbc_outer = Dirichlet(:u, ∂outer, (x,t) -> 0)
        mean_value_constraint = setup_mean_constraint(cellvalues, dh)
        add!(dbc, dbc_outer);
        add!(dbc, mean_value_constraint);
    end
    close!(dbc)
    return dbc
end;

#divergence of Diffusion matrix 
function my_divergence(D::Matrix{Function}, v::Vec{2, Float64})
    gA_11 = Float64.(gradient(D[1,1], v))
    gA_12 = Float64.(gradient(D[1,2], v))
    gA_21 = Float64.(gradient(D[2,1], v))
    gA_22 = Float64.(gradient(D[2,2], v))

    return Vec(gA_11[1] + gA_12[2],  gA_21[1] + gA_22[2])
end

#laplacian for Diffusion matrix 
function my_laplace(D::Matrix{Function}, v::Vec{2, Float64})
    gA_11 = laplace(D[1,1], v)
    gA_12 = laplace(D[1,2], v)
    gA_21 = laplace(D[2,1], v)
    gA_22 = laplace(D[2,2], v)
    return gA_11 + gA_12+  gA_21 +  gA_22
end

#divergence for drift
function my_divergence_drift(g::Vector{Function}, v::Vec{2, Float64})
    gA_11 = Float64.(gradient(g[1], v))
    gA_12 = Float64.(gradient(g[2], v))

    return gA_11[1] + gA_12[2]
end

#constraint to enforce ∫μ = 1
function setup_mean_constraint(cellvalues::CellScalarValues, dh)
    assembler = start_assemble()
    range_u = dof_range(dh, :u)
    element_dofs = zeros(Int, ndofs_per_cell(dh))
    element_dofs_u = view(element_dofs, range_u)
    element_coords = zeros(Vec{2}, 3)
    Ce = zeros(1, length(range_u)) 

    for (cell_num, cell) in enumerate(CellIterator(dh))
        Ce .= 0
        getcoordinates!(element_coords, dh.grid, cell_num)
        print()
        reinit!(cellvalues, cell)
        celldofs!(element_dofs, dh, cell_num)
        for qp in 1:getnquadpoints(cellvalues)
            dΓ = getdetJdV(cellvalues, qp)
            for i in 1:getnbasefunctions(cellvalues)
                Ce[1, i] += shape_value(cellvalues, qp, i) * dΓ
            end
        end
        # Assemble to row 1
        assemble!(assembler, [1], element_dofs_u, Ce)
    end

    C = end_assemble(assembler)
    # Create an AffineConstraint from the C-matrix
    _, J, V = findnz(C)
    _, constrained_dof_idx = findmax(abs2, V)
    constrained_dof = J[constrained_dof_idx]
    #val = V[constrained_dof_idx]
    #V ./= V[constrained_dof_idx]
    mean_value_constraint = AffineConstraint(
        constrained_dof,
        Pair{Int,Float64}[J[i] => -V[i] for i in 1:length(J) if J[i] != constrained_dof],
        1.0,
    )
    return mean_value_constraint
end


#forward commonitor function , done!
function assemble_element_forward!(Ke::Matrix, fe::Vector, base_info::grid_info, 
    coords::Vector{Vec{2, Float64}})
    cellvalues = base_info.cellvalues_forward
    g = base_info.g
    D = base_info.D
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        coords_qp = spatial_coordinate(cellvalues, q_point, coords)
        drift = eval_g(g, coords_qp) # -∇V
        D_true = eval_D(D, coords_qp)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ * 0
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (-0.5 * ∇δu⋅(D_true*∇u) - (-drift + 0.5*my_divergence(D, coords_qp)) ⋅ ∇u * δu) * dΩ
            end
        end
    end
    return Ke, fe
end


#this code implement the stationary distribution for the whole system in the original grid
function assemble_element_stationary!(Ke::Matrix, fe::Vector, base_info::grid_info, 
    coords::Vector{Vec{2, Float64}})
    cellvalues = base_info.cellvalues_stat
    D = base_info.D
    g = base_info.g
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        coords_qp = spatial_coordinate(cellvalues, q_point, coords) # spatial coordinates 
        drift = eval_g(g, coords_qp) # -∇V
        div_drift = my_divergence_drift(g, coords_qp) # ∇⋅b = ∇⋅-∇V = -ΔV
        D_div  = my_divergence(D,coords_qp) # ∇⋅D
        D_laplace = my_laplace(D,coords_qp) # ΔD

        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ * 0
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                u = shape_value(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (-0.5 * ∇δu⋅(eval_D(D, coords_qp)*∇u) + (u * (-div_drift + 0.5*D_laplace)) * δu +
                                  ((-drift+0.5*D_div) ⋅ ∇u)* δu) * dΩ
            end
        end
    end
    return Ke, fe
end


#assemble function 
function assemble_global(K::SparseMatrixCSC, base_info::grid_info, react_stat::reactive_stat, type::String)
    if type == "forward"
        cellvalues = base_info.cellvalues_forward
        dh = base_info.dh_comm
    elseif type == "backward"
        cellvalues = base_info.cellvalues_backward
        dh = base_info.dh_comm
    else
        cellvalues = base_info.cellvalues_stat
        dh = base_info.dh_stat
    end
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cels
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        coords = getcoordinates(cell)
        if type == "forward"
            assemble_element_forward!(Ke, fe, base_info ,coords)
        elseif type == "stationary"
            assemble_element_stationary!(Ke, fe, base_info, coords)
        else
            assemble_element_backward!(Ke, fe, base_info, coords, react_stat)
        end
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

#backward commonitor function # this part needs some careful manipulation with stationary distribution 
function assemble_element_backward!(Ke::Matrix, fe::Vector, base_info::grid_info, 
    coords::Vector{Vec{2, Float64}}, react_stat::reactive_stat)
    # information from base_info
    g = base_info.g
    #c = base_info.c
    D = base_info.D
    cellvalues = base_info.cellvalues_backward
    dh_stat = base_info.dh_stat
    #information from reactive statistics
    grad_μ_projected = react_stat.grad_μ_projected # grad μ
    projector = react_stat.grad_μ_projector # projector 
    μ_projected = react_stat.u_stat

    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        coords_qp = spatial_coordinate(cellvalues, q_point, coords)
        drift = eval_g(g, coords_qp)

        ph = PointEvalHandler(base_info.grid_stat, [coords_qp], warn = false)
        μ = Ferrite.get_point_values(ph, dh_stat, μ_projected, :u);
        ∇μ = get_point_values(ph, projector, grad_μ_projected);

        D_div_part = my_divergence(D,coords_qp) + eval_D(D, coords_qp) * (∇μ./μ)[1]
        if isnan(D_div_part[1]) || isnan(D_div_part[2])
            D_div_part = my_divergence(D,coords_qp)
        end
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ * 0
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (-0.5 * ∇δu⋅(eval_D(D, coords_qp)*∇u) -
                            (drift + 0.5*my_divergence(D, coords_qp)) ⋅ ∇u * δu + D_div_part ⋅ ∇u * δu) * dΩ
            end
        end
    end
    return Ke, fe
end


#compute the reactive density of this process by 1/zμq+q- # again this part needs some careful manipulation with stationary μ
function compute_reaction_density(base_info::grid_info, react_stat::reactive_stat) 
    #initialize parameters from grid_info
    cellvalues = base_info.cellvalues_forward;
    dh = base_info.dh_comm;
    q_backward = react_stat.u_backward;
    q_forward = react_stat.u_forward;
    ####
    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)
    # Allocate storage for the fluxes to store
    q = [Vec{1, Float64}[] for _ in 1:getncells(dh.grid)]
    Z_normalization = 0; #normalization constant 
    for (cell_num, cell) in enumerate(CellIterator(dh))
        coords = getcoordinates(cell)
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        q_backwardᵉ = q_backward[cell_dofs]
        q_forwardᵉ = q_forward[cell_dofs]
        reinit!(cellvalues, cell)
        for q_point in 1:nqp
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            forward_value = function_value(cellvalues, q_point, q_forwardᵉ) #forward communitor value 
            backward_value = function_value(cellvalues, q_point, q_backwardᵉ) #backward communitor value 
            ph = PointEvalHandler(base_info.grid_stat, [coords_qp], warn = false)
            μ_value = Ferrite.get_point_values(ph, base_info.dh_stat, react_stat.u_stat, :u);# stationary density value 
            q_qp = forward_value*backward_value*μ_value[1]
            Z_normalization += q_qp * dΩ
            push!(q_cell, Vec(q_qp))
        end
    end
    return q, Z_normalization 
end



# compute the reactive current for this reactive path as well as the transition rate
function compute_reaction_current(base_info::grid_info, react_stat::reactive_stat)
    cellvalues = base_info.cellvalues_forward
    dh = base_info.dh_comm
    q_backward = react_stat.u_backward
    D = base_info.D
    g = base_info.g
    q_forward = react_stat.u_forward #forward commonitor for computing transition rate 
    grad_μ_projected = react_stat.grad_μ_projected # grad μ
    projector = react_stat.grad_μ_projector # projector 
    μ_projected = react_stat.u_stat
    transition_rate = 0 #transition rate 
    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)
    # Allocate storage for the fluxes to store
    q = [Vec{2,Float64}[] for _ in 1:getncells(dh.grid)]
    for (cell_num, cell) in enumerate(CellIterator(dh))
        coords = getcoordinates(cell)
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        q_backwardᵉ = q_backward[cell_dofs]
        q_forwardᵉ = q_forward[cell_dofs]
        reinit!(cellvalues, cell)
        for q_point in 1:nqp
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)# coordinates 
            ph = PointEvalHandler(base_info.grid_stat, [coords_qp], warn = false)
            μ = Ferrite.get_point_values(ph, base_info.dh_stat, μ_projected, :u); #μ 
            ∇μ = get_point_values(ph, projector, grad_μ_projected); #∇μ 
            value = function_value(cellvalues, q_point, q_backwardᵉ) #backward communitor value 
            grad_value = function_gradient(cellvalues, q_point, q_backwardᵉ)# gradient of backward communitor 
            value_forward = function_value(cellvalues, q_point, q_forwardᵉ) #forward communitor value 
            grad_value_forward = function_gradient(cellvalues, q_point, q_forwardᵉ)# gradient of forward communitor
            J = eval_g(g, coords_qp) * μ[1] .- 0.5* μ[1] * my_divergence(D, coords_qp) - 0.5*eval_D(D, coords_qp)*∇μ[1]
            q_qp = J*value*value_forward + 0.5*μ[1]*eval_D(D, coords_qp)*(value*grad_value_forward - value_forward*grad_value)
            push!(q_cell, Vec(q_qp[1], q_qp[2]))
            transition_rate += μ[1] * Vector(grad_value_forward)' * eval_D(D, coords_qp) ⋅ grad_value_forward
        end
    end
    return q, transition_rate
end


# determining the value of diffusion matrix D at coordinates x
function eval_D(D, x)
    return [D[1,1](x) D[1,2](x); D[2,1](x) D[2,2](x)]
end

#determining the value of the drift as a vector 
function eval_g(g, x)
    return Vec(g[1](x), g[2](x))
end

# compute ∇μ and do the projection 
function compute_grad_μ(cellvalues::CellScalarValues{dim,T}, dh::DofHandler, u, grid) where {dim,T}
    n = getnbasefunctions(cellvalues) # number of basis functions 
    cell_dofs = zeros(Int, n) # cell degree of freedoms
    nqp = getnquadpoints(cellvalues) # number of nodes on each cell 
    # Allocate storage for the fluxes to store
    q = [Vec{2,T}[] for _ in 1:getncells(dh.grid)] 
    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        q_ᵉ = u[cell_dofs]
        reinit!(cellvalues, cell)
        for q_point in 1:nqp
            q_qp = function_gradient(cellvalues, q_point, q_ᵉ) 
            push!(q_cell, q_qp)
        end
    end
    projector = L2Projector(Lagrange{dim, RefTetrahedron, 1}(), grid);
    grad_μ_projected = Ferrite.project(projector, q, QuadratureRule{dim, RefTetrahedron}(2); project_to_nodes=false);
    return projector, grad_μ_projected
end

# compute ∇μ and do the projection 
function compute_grad_μ(base_info::grid_info, u)
    cellvalues = base_info.cellvalues_stat;
    dh = base_info.dh_stat;
    grid = base_info.grid_stat;
    dim = 2
    n = getnbasefunctions(cellvalues) # number of basis functions 
    cell_dofs = zeros(Int, n) # cell degree of freedoms
    nqp = getnquadpoints(cellvalues) # number of nodes on each cell 
    # Allocate storage for the fluxes to store
    q = [Vec{2,Float64}[] for _ in 1:getncells(dh.grid)] 
    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        q_ᵉ = u[cell_dofs]
        reinit!(cellvalues, cell)
        for q_point in 1:nqp
            q_qp = function_gradient(cellvalues, q_point, q_ᵉ) 
            push!(q_cell, q_qp)
        end
    end
    projector = L2Projector(Lagrange{dim, RefTetrahedron, 1}(), grid);
    grad_μ_projected = Ferrite.project(projector, q, QuadratureRule{dim, RefTetrahedron}(2); project_to_nodes=false);
    return projector, grad_μ_projected
end

#compute the transition rate of the reactive process 
function compute_transition_rate(base_info::grid_info, react_stat::reactive_stat)
    cellvalues = base_info.cellvalues_forward
    dh = base_info.dh_comm
    q_backward = react_stat.u_backward
    D = base_info.D
    g = base_info.g

    grad_μ_projected = react_stat.grad_μ_projected # grad μ
    projector = react_stat.grad_μ_projector # projector 
    μ_projected = react_stat.u_stat

    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)
    # Allocate storage for the fluxes to store
    q = [Vec{2,Float64}[] for _ in 1:getncells(dh.grid)]
    for (cell_num, cell) in enumerate(CellIterator(dh))
        coords = getcoordinates(cell)
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        q_backwardᵉ = q_backward[cell_dofs]
        reinit!(cellvalues, cell)
        for q_point in 1:nqp
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)# coordinates 
            ph = PointEvalHandler(base_info.grid_stat, [coords_qp], warn = false)
            μ = Ferrite.get_point_values(ph, base_info.dh_stat, μ_projected, :u); #μ 
            ∇μ = get_point_values(ph, projector, grad_μ_projected); #∇μ 
            value = function_value(cellvalues, q_point, q_backwardᵉ) #communitor value 
            grad_value = function_gradient(cellvalues, q_point, q_backwardᵉ)# gradient of communitor 

            q_qp = -gradient(g, coords_qp) * μ[1] * value
                    .- 0.5*μ[1] * value * my_divergence(D, coords_qp)
                        -0.5* eval_D(D, coords_qp)*(∇μ[1] * value .+ grad_value * μ[1]) 
            push!(q_cell, q_qp)
        end
    end
    return q

end

#rectangle mesh for commonitor with selected circle
function create_rect_mesh_comm(A_center, A_radius, B_center, B_radius, left_bott,left_upper, right_bott, right_upper)
    gmsh.initialize()
    dim = Int64(gmsh.model.getDimension())
    gmsh.model.occ.addPoint(left_bott[1], left_bott[2], 0, 0, 1)
    gmsh.model.occ.addPoint(left_upper[1], left_upper[2], 0, 0, 2)
    gmsh.model.occ.addPoint(right_bott[1], right_bott[2], 0, 0, 3)
    gmsh.model.occ.addPoint(right_upper[1], right_upper[2], 0, 0, 4)

    gmsh.model.occ.addLine(1, 2, 5)
    gmsh.model.occ.addLine(3, 1, 6)
    gmsh.model.occ.addLine(2, 4, 7)
    gmsh.model.occ.addLine(4, 3, 8)

    gmsh.model.occ.addCurveLoop([5, 6, 7, 8], 1)

    gmsh.model.occ.addPlaneSurface([1], 1)

    hole1 = gmsh.model.occ.addDisk(A_center[1], A_center[2], 0, A_radius, A_radius)
    hole2 = gmsh.model.occ.addDisk(B_center[1], B_center[2], 0, B_radius, B_radius)
    membrane1 = gmsh.model.occ.cut([(2, 1)], [(2, hole1), (2,hole2)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities()
    gdim = 2
    gmsh.model.addPhysicalGroup(volumes[1][1], [volumes[1][2]], 11)

    boundary = gmsh.model.getBoundary(membrane1[1])
    boundary_ids = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(2, boundary_ids)

    meshSize = 0.05
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",meshSize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",meshSize)
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

    addfaceset!(grid, "top", x -> x[2] ≈ left_upper[2]);
    addfaceset!(grid, "left", x -> x[1] ≈ left_upper[1]);
    addfaceset!(grid, "right", x -> x[1] ≈ right_upper[1]);
    addfaceset!(grid, "bottom", x -> x[2] ≈ right_bott[2]);
    addfaceset!(grid, "A", x -> (x[1]-A_center[1])^2+(x[2]-A_center[2])^2 ≈ A_radius^2);
    addfaceset!(grid, "B", x -> (x[1]-B_center[1])^2+(x[2]-B_center[2])^2 ≈ B_radius^2);
    return grid        
end

# rect mesh for stationary distribution 
function create_rect_mesh(left_bott,left_upper, right_bott, right_upper)
    gmsh.initialize()
    dim = Int64(gmsh.model.getDimension())
    
    gmsh.model.occ.addPoint(left_bott[1], left_bott[2], 0, 0, 1)
    gmsh.model.occ.addPoint(left_upper[1], left_upper[2], 0, 0, 2)
    gmsh.model.occ.addPoint(right_bott[1], right_bott[2], 0, 0, 3)
    gmsh.model.occ.addPoint(right_upper[1], right_upper[2], 0, 0, 4)

    gmsh.model.occ.addLine(1, 2, 5)
    gmsh.model.occ.addLine(3, 1, 6)
    gmsh.model.occ.addLine(2, 4, 7)
    gmsh.model.occ.addLine(4, 3, 8)

    gmsh.model.occ.addCurveLoop([5, 6, 7, 8], 1)

    rect = gmsh.model.occ.addPlaneSurface([1], 1)
    
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [rect], 2)
    
    meshSize = 0.05
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",meshSize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",meshSize)
    gdim = 2
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.reverse()
    dim = Int64(gmsh.model.getDimension())

    facedim = dim - 1 

    nodes = tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    # "Domain" is the name of a PhysicalGroup and saves all cells that define the computational domain
    domaincellset = cellsets["2"]
    elements = elements[collect(domaincellset)]

    boundarydict = toboundary(facedim)
    facesets = tofacesets(boundarydict, elements)
    gmsh.finalize()

    grid = Grid(elements, nodes, facesets=facesets, cellsets=cellsets)
    addfaceset!(grid, "top", x -> x[2] ≈ left_upper[2]);
    addfaceset!(grid, "left", x -> x[1] ≈ left_upper[1]);
    addfaceset!(grid, "right", x -> x[1] ≈ right_upper[1]);
    addfaceset!(grid, "bottom", x -> x[2] ≈ right_bott[2]);
    return grid
end


#find the center and radius of stable states such that the difference in energy is less than "level" 
function findcenter_radius(st::Vector{Float64}, ed::Vector{Float64}, base_info::grid_info, react_stat::reactive_stat, level = 0.25)
    length = 100
    points = [Vec((x, y)) for y in range(st[2], ed[2], length=length) for x in range(st[1], ed[1],length = length)];
    ph_stat = PointEvalHandler(base_info.grid_stat, points, warn = false);
    u_points_stat = Ferrite.get_point_values(ph_stat, base_info.dh_stat, react_stat.u_stat, :u);
    #center with highest probability
    index = argmax(u_points_stat)
    val = u_points_stat[index]
    center = points[index]
    radius = 0.25
    return center, radius
end

#compute the forward, backward committor function, transition density, transition current 
function obtain_commonitor(base_info::grid_info, react_stat::reactive_stat)
    # assembly and solve (compute the forward commonitor function)
    K = create_sparsity_pattern(base_info.dh_comm);
    K, f = assemble_global(K, base_info, react_stat, "forward");
    apply!(K, f, base_info.dbc_for)
    u_forward = K \ f;
    react_stat.u_forward = u_forward;
    #now compute the gradient of stationary density function and use it 
    #with stationary distribution to compute the backward commonitor function 
    projector, grad_μ_projected = compute_grad_μ(base_info, react_stat.u_stat);
    react_stat.grad_μ_projected = grad_μ_projected;
    react_stat.grad_μ_projector = projector;

    #backward commonitor function 
    K = create_sparsity_pattern(base_info.dh_comm);
    K, f = assemble_global(K, base_info, react_stat, "backward");
    apply!(K, f, base_info.dbc_back)
    u_backward = K \ f;
    react_stat.u_backward = u_backward;

    #determining the stationary distribution by μAB = μ*q⁺q⁻, and normalization constant ∫μ*q⁺q⁻
    distribution_gp, Z_normalization = compute_reaction_density(base_info, react_stat);
    projector = L2Projector(base_info.ip, base_info.grid_comm);
    distribution_projected = Ferrite.project(projector, distribution_gp, QuadratureRule{2, RefTetrahedron}(2); project_to_nodes=false)./Z_normalization;
    react_stat.reactive_density = distribution_projected;
    react_stat.Z_normalization = Z_normalization;
    #determining the current of reactive path JAB
    current_gp, transition_rate = compute_reaction_current(base_info, react_stat);
    projector = L2Projector(base_info.ip, base_info.grid_comm);
    current_projected = Ferrite.project(projector, current_gp, QuadratureRule{2, RefTetrahedron}(2); project_to_nodes=false);
    react_stat.reactive_current = current_projected;
    react_stat.transition_rate = transition_rate;
end

# compute the stationary density function 
function compute_stat_dist(base_info::grid_info, react_stat::reactive_stat)
    K = create_sparsity_pattern(base_info.dh_stat, base_info.dbc_stats);
    K, f = assemble_global(K, base_info, react_stat, "stationary");
    apply!(K, f, base_info.dbc_stats);
    u_stat = K \ f;
    apply!(u_stat, base_info.dbc_stats);
    react_stat.u_stat = u_stat;
end

#obtain the spline for boundary of set A and set B
function obtain_spl(base_info, react_stat, st_A, ed_A)
    len = 100
    points = [Vec((round(x, digits = 2), round(y, digits = 2))) for y in range(st_A[2], ed_A[2], length=len) for x in range(st_A[1], ed_A[1],length = len)];
    ph_stat = PointEvalHandler(base_info.grid_stat, points, warn = false);
    u_points_stat = round.(Ferrite.get_point_values(ph_stat, base_info.dh_stat, react_stat.u_stat, :u),digits = 2);
    #center with highest probability
    index = argmax(u_points_stat)
    val = u_points_stat[index]
    center = points[index]
    target_val = round(exp(log(val) - 0.25), digits = 2)
    target = points[u_points_stat .≈ target_val]
    lower = target[findall(x->x[2] < center[2], target)];lower = permutedims(hcat(lower...))
    upper = target[findall(x->x[2] > center[2], target)];upper = permutedims(hcat(upper...))

    lower =lower[sortperm(lower[:, 1]), :]
    upper= upper[sortperm(upper[:, 1], rev = true), :]

    lower = [eachrow(lower)...];upper = [eachrow(upper)...];
    hull = convex_hull(lower); lower = VPolygon(hull).vertices; lower = permutedims(hcat(lower...)); lower = lower[sortperm(lower[:, 1]), :]
    hull = convex_hull(upper); upper = VPolygon(hull).vertices; upper = permutedims(hcat(upper...)); upper = upper[sortperm(upper[:, 1], rev = true), :]
    return lower, upper, target_val
end

#check function for adding the face value of the boundary
function obtain_face_boundary(x, base_info, react_stat, st, ed, target_val)
    points = [Vec(round(x[1], digits = 2), round(x[2], digits = 2))]
    check = false
    if x[1] >= st[1] && x[2] >= st[2] && x[1] <= ed[1] && x[2] <=ed[2]
        check = true
    end
    ph_stat = PointEvalHandler(base_info.grid_stat, points, warn = false);
    u_points_stat = round.(Ferrite.get_point_values(ph_stat, base_info.dh_stat, react_stat.u_stat, :u),digits = 2);
    return u_points_stat[1] ≈ target_val && check
end
