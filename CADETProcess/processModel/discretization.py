"""
================================================================
Discretization (:mod:`CADETProcess.processModel.discretization`)
================================================================

The following module is a collection of classes that define different discretization
schemes for :mod:`UnitOperations <CADETProcess.processModel.unitOperation>` for the
CADETProcess software package.

The :class:`~CADETProcess.processModel.DiscretizationParametersBase` class is the base
class for all other classes in this module and defines some common parameters.
Specific parameters for each scheme are defined as attributes of each class.

"""
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Bool, Switch,
    RangedInteger, UnsignedInteger, UnsignedFloat,
    SizedRangedList
)


__all__ = [
    'NoDiscretization',
    'LRMDiscretizationFV', 'LRMDiscretizationDG',
    'LRMPDiscretizationFV', 'LRMPDiscretizationDG',
    'GRMDiscretizationFV', 'GRMDiscretizationDG',
    'WenoParameters', 'ConsistencySolverParameters',
    'DGMixin'
]


class DiscretizationParametersBase(Structure):
    """Base class for storing discretization parameters.

    Attributes
    ----------
    dimensionality : dict
        Dimensionality of the parameters
    weno_parameters : WenoParameters
        Parameters for the WENO scheme.
    consistency_solver: ConsistencySolverParameters
        Consistency solver parameters for Cadet.

    """

    _dimensionality = []

    def __init__(self):
        """Initialize a new DiscretizationParametersBase instance."""
        self.weno_parameters = WenoParameters()
        self.consistency_solver = ConsistencySolverParameters()

        super().__init__()

    @property
    def dimensionality(self):
        """dict: Dimensionality of the parameters."""
        dim = {}
        for d in self._dimensionality:
            v = getattr(self, d)
            if v is None:
                continue
            dim[d] = v

        return dim

    @property
    def parameters(self):
        """dict: Dictionary with parameter values."""
        parameters = super().parameters
        parameters['weno'] = self.weno_parameters.parameters
        parameters['consistency_solver'] = self.consistency_solver.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.weno_parameters.parameters = parameters.pop('weno')
        except KeyError:
            pass
        try:
            self.consistency_solver.parameters \
                = parameters.pop('consistency_solver')
        except KeyError:
            pass

        super(DiscretizationParametersBase, self.__class__).parameters.fset(
            self, parameters
        )


class NoDiscretization(DiscretizationParametersBase):
    """Class for unit operations without spatial discretization."""

    pass


class DGMixin(DiscretizationParametersBase):
    pass


class LRMDiscretizationFV(DiscretizationParametersBase):
    """Discretization parameters of the FV version of the LRM.

    This class stores parameters for the Lax-Richtmyer-Morton (LRM) flux-based
    finite volume discretization.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 100.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.

    """

    ncol = UnsignedInteger(default=100)
    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'use_analytic_jacobian', 'reconstruction',
    ]
    _dimensionality = ['ncol']


class LRMDiscretizationDG(DGMixin):
    """Discretization parameters of the DG version of the LRM.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 16.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.
    polynomial_degree : UnsignedInteger, optional
        Degree of the polynomial used for axial discretization. Default is 3.
    polydeg : UnsignedInteger, optional
        Degree of the polynomial used for axial discretization. Default is 3.
    exact_integration : Bool, optional
        Whether to use exact integration for the axial discretization.
        Default is False.

    See Also
    --------
    CADETProcess.processModel.LRMPDiscretizationFV
    CADETProcess.processModel.LumpedRateModelWithPores

    """

    ncol = UnsignedInteger(default=16)
    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    polynomial_degree = UnsignedInteger(default=3)
    polydeg = polynomial_degree
    exact_integration = Bool(default=False)

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'use_analytic_jacobian', 'reconstruction',
        'polydeg', 'exact_integration'
    ]
    _dimensionality = ['axial_dof']

    @property
    def axial_dof(self):
        """int: Number of degrees of freedom in the axial discretization."""
        return self.ncol * (self.polynomial_degree + 1)


class LRMPDiscretizationFV(DiscretizationParametersBase):
    """Discretization parameters of the FV version of the LRMP.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 100.
    par_geom : Switch, optional
        The geometry of the particles in the model.
        Valid values are 'SPHERE', 'CYLINDER', and 'SLAB'.
        Default is 'SPHERE'.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.
    gs_type : Bool, optional
        Type of Gram-Schmidt orthogonalization.
        If 0, use classical Gram-Schmidt.
        If 1, use modified Gram-Schmidt.
        The default is 1.
    max_krylov : UnsignedInteger, optional
        Size of the Krylov subspace in the iterative linear GMRES solver.
        If 0, max_krylov = NCOL * NCOMP * NPARTYPE is used.
        The default is 0.
    max_restarts : UnsignedInteger, optional
        Maximum number of restarts to use for the GMRES method. Default is 10.
    schur_safety : UnsignedFloat, optional
        Safety factor for the Schur complement solver. Default is 1.0e-8.

    See Also
    --------
    CADETProcess.processModel.LRMPDiscretizationDG
    CADETProcess.processModel.LumpedRateModelWithPores

    """

    ncol = UnsignedInteger(default=100)

    par_geom = Switch(
        default='SPHERE',
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])

    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'par_geom',
        'use_analytic_jacobian', 'reconstruction',
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety'
    ]
    _dimensionality = ['ncol']


class LRMPDiscretizationDG(DGMixin):
    """Discretization parameters of the DG version of the LRMP.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 16.
    par_geom : Switch, optional
        The geometry of the particles in the model.
        Valid values are 'SPHERE', 'CYLINDER', and 'SLAB'.
        Default is 'SPHERE'.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.
    polynomial_degree : UnsignedInteger, optional
        Degree of the polynomial used for spatial discretization. Default is 3.
    polydeg : UnsignedInteger, optional
        Alias for polynomial_degree.
    exact_integration : Bool, optional
        Whether to use exact integration for the spatial discretization.
        Default is False.
    gs_type : Bool, optional
        Type of Gram-Schmidt orthogonalization.
        If 0, use classical Gram-Schmidt.
        If 1, use modified Gram-Schmidt.
        The default is 1.
    max_krylov : UnsignedInteger, optional
        Size of the Krylov subspace in the iterative linear GMRES solver.
        If 0, max_krylov = NCOL * NCOMP * NPARTYPE is used.
        The default is 0.
    max_restarts : UnsignedInteger, optional
        Maximum number of restarts to use for the GMRES method. Default is 10.
    schur_safety : UnsignedFloat, optional
        Safety factor for the Schur complement solver. Default is 1.0e-8.

    See Also
    --------
    CADETProcess.processModel.LRMPDiscretizationFV
    CADETProcess.processModel.LumpedRateModelWithPores

    """

    ncol = UnsignedInteger(default=16)

    par_geom = Switch(
        default='SPHERE',
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    polynomial_degree = UnsignedInteger(default=3)
    polydeg = polynomial_degree
    exact_integration = Bool(default=False)

    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'par_geom',
        'use_analytic_jacobian', 'reconstruction',
        'polydeg', 'exact_integration',
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety'
    ]
    _dimensionality = ['axial_dof']

    @property
    def axial_dof(self):
        """int: Number of axial degrees of freedom in the spatial discretization."""
        return self.ncol * (self.polynomial_degree + 1)


class GRMDiscretizationFV(DiscretizationParametersBase):
    """Discretization parameters of the FV version of the LRMP.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 100.
    npar : UnsignedInteger, optional
        Number of discretization cells in the radial direction. Default is 5.
    par_geom : Switch, optional
        The geometry of the particles in the model.
        Valid values are 'SPHERE', 'CYLINDER', and 'SLAB'.
        Default is 'SPHERE'.
    par_disc_type : Switch, optional
        Discretization scheme inside the particles for all or each particle type.
        Valid values are 'EQUIDISTANT_PAR', 'EQUIVOLUME_PAR', and 'USER_DEFINED_PAR'.
        Default is 'EQUIDISTANT_PAR'.
    par_disc_vector : SizedRangedList, optional
        Node coordinates for the cell boundaries
        (ignored if `par_disc_type` != `USER_DEFINED_PAR).
        The coordinates are relative and have to include the endpoints `0` and `1`.
        They are later linearly mapped to the true radial range.
        The coordinates for each particle type are appended to one long vector in
        type-major ordering.
        Default is a uniformly spaced vector with `npar+1` points between 0 and 1.
    par_boundary_order : RangedInteger, optional
        The order of the boundary scheme used to discretize the particles.
        Valid values are 1 (first order) and 2 (second order).
        Default is 2.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.
    gs_type : Bool, optional
        Type of Gram-Schmidt orthogonalization.
        If 0, use classical Gram-Schmidt.
        If 1, use modified Gram-Schmidt.
        The default is 1.
    max_krylov : UnsignedInteger, optional
        Size of the Krylov subspace in the iterative linear GMRES solver.
        If 0, max_krylov = ncol * ncomp * npar is used, where ncomp is the
        number of components in the model.
        Default is 0.
    max_restarts : UnsignedInteger, optional
        Maximum number of restarts to use for the GMRES method. Default is 10.
    schur_safety : UnsignedFloat, optional
        Safety factor for the Schur complement solver. Default is 1.0e-8.
    fix_zero_surface_diffusion : Bool, optional
        If True, fix the surface diffusion coefficient of particles with zero
        surface diffusion to a small positive value. Default is False.

    See Also
    --------
    CADETProcess.processModel.LRMPDiscretizationDG
    CADETProcess.processModel.LumpedRateModelWithPores

    """

    ncol = UnsignedInteger(default=100)
    npar = UnsignedInteger(default=5)

    par_geom = Switch(
        default='SPHERE',
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )
    par_disc_type = Switch(
        default='EQUIDISTANT_PAR',
        valid=['EQUIDISTANT_PAR', 'EQUIVOLUME_PAR', 'USER_DEFINED_PAR']
    )
    par_disc_vector = SizedRangedList(
        lb=0, ub=1, size='par_disc_vector_length', is_optional=True
    )

    par_boundary_order = RangedInteger(lb=1, ub=2, default=2)

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])

    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)

    fix_zero_surface_diffusion = Bool(default=False)

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'npar',
        'par_geom', 'par_disc_type', 'par_disc_vector', 'par_boundary_order',
        'use_analytic_jacobian', 'reconstruction',
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety',
        'fix_zero_surface_diffusion',
    ]
    _required_parameters = [
        'ncol', 'npar',
        'par_geom', 'par_disc_type', 'par_boundary_order',
        'use_analytic_jacobian', 'reconstruction',
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety',
        'fix_zero_surface_diffusion',
    ]
    _dimensionality = ['ncol', 'npar']

    @property
    def par_disc_vector_length(self):
        """int: Number of entries in the particle discretization vector."""
        return self.npar + 1


class GRMDiscretizationDG(DGMixin):
    """Discretization parameters of the DG version of the GRM.

    Attributes
    ----------
    ncol : UnsignedInteger, optional
        Number of axial column discretization cells. Default is 16.
    npar : UnsignedInteger, optional
        Number of particle (radial) discretization cells for each particle type.
        Default is 1.
    nparcell : UnsignedInteger, optional
        Alias for npar
    par_geom : Switch, optional
        The geometry of the particles in the model.
        Valid values are 'SPHERE', 'CYLINDER', and 'SLAB'.
        Default is 'SPHERE'.
    use_analytic_jacobian : Bool, optional
        If True, use analytically computed Jacobian matrix (faster).
        If False, use Jacobians generated by algorithmic differentiation (slower).
        Default is True.
    reconstruction : Switch, optional
        Method for spatial reconstruction. Valid values are 'WENO' (Weighted
        Essentially Non-Oscillatory). Default is 'WENO'.
    polynomial_degree : UnsignedInteger, optional
        Degree of the polynomial used for axial discretization. Default is 3.
    polydeg : UnsignedInteger, optional
        Alias for polynomial_degree.
    polynomial_degree_particle : UnsignedInteger, optional
        Degree of the polynomial used for particle radial discretization. Default is 3.
    exact_integration : Bool, optional
        Whether to use exact integration for the axial discretization.
        Default is False.
    exact_integration : Bool, optional
        Whether to use exact integration for the particle radial discretization.
        Default is False.
    gs_type : Bool, optional
        Type of Gram-Schmidt orthogonalization.
        If 0, use classical Gram-Schmidt.
        If 1, use modified Gram-Schmidt.
        The default is 1.
    max_krylov : UnsignedInteger, optional
        Size of the Krylov subspace in the iterative linear GMRES solver.
        If 0, max_krylov = NCOL * NCOMP * NPARTYPE is used.
        The default is 0.
    max_restarts : UnsignedInteger, optional
        Maximum number of restarts to use for the GMRES method. Default is 10.
    schur_safety : UnsignedFloat, optional
        Safety factor for the Schur complement solver. Default is 1.0e-8.
    fix_zero_surface_diffusion : Bool, optional
        Whether to fix zero surface diffusion for particles. Default is False.
        If True, the parameters must not become non-zero during this or subsequent
        simulation runs. The internal data structures are optimized for a more efficient
        simulation.
        Default is False (optimization disabled in favor of flexibility).

    See Also
    --------
    CADETProcess.processModel.GRMDiscretizationDG
    CADETProcess.processModel.GeneralRateModel

    """

    ncol = UnsignedInteger(default=16)
    npar = UnsignedInteger(default=1)
    nparcell = npar

    par_geom = Switch(
        default='SPHERE',
        valid=['SPHERE', 'CYLINDER', 'SLAB']
    )
    par_disc_type = Switch(
        default='EQUIDISTANT_PAR',
        valid=['EQUIDISTANT_PAR', 'EQUIVOLUME_PAR', 'USER_DEFINED_PAR']
    )
    par_disc_vector = SizedRangedList(
        lb=0, ub=1, size='par_disc_vector_length'
    )

    par_boundary_order = RangedInteger(lb=1, ub=2, default=2)

    use_analytic_jacobian = Bool(default=True)
    reconstruction = Switch(default='WENO', valid=['WENO'])
    polynomial_degree = UnsignedInteger(default=3)
    polydeg = polynomial_degree
    polynomial_degree_particle = UnsignedInteger(default=3)
    parpolydeg = polynomial_degree_particle
    exact_integration = Bool(default=False)
    exact_integration_particle = Bool(default=True)
    par_exact_integration = exact_integration_particle

    gs_type = Bool(default=True)
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1.0e-8)

    fix_zero_surface_diffusion = Bool(default=False)

    _parameters = DiscretizationParametersBase._parameters + [
        'ncol', 'nparcell',
        'par_geom', 'par_disc_type', 'par_disc_vector', 'par_boundary_order',
        'use_analytic_jacobian', 'reconstruction',
        'polydeg', 'parpolydeg',
        'exact_integration', 'par_exact_integration',
        'gs_type', 'max_krylov', 'max_restarts', 'schur_safety',
        'fix_zero_surface_diffusion',
    ]
    _dimensionality = ['axial_dof', 'par_dof']

    @property
    def axial_dof(self):
        """int: Number of axial degrees of freedom in the axial discretization."""
        return self.ncol * (self.polynomial_degree + 1)

    @property
    def par_dof(self):
        """int: Number of particle degrees of freedom in the axial discretization."""
        return self.ncol * (self.polynomial_degree_particle + 1)

    @property
    def par_disc_vector_length(self):
        """int: Number of entries in the particle discretization vector."""
        return self.npar + 1


class WenoParameters(Structure):
    """Discretization parameters for the WENO scheme.

    Attributes
    ----------
    boundary_model : Switch, optional
        Specifies the method for dealing with boundary cells.
        Valid values are:
        0: Lower WENO order (stable)
        1: Zero weights (unstable for small `D_ax`)
        2: Zero weights for p != 0 (stable?)
        3: Large ghost points
        Default is 0.
    weno_eps : UnsignedFloat, optional
        A small positive number used to avoid division by zero in the WENO scheme.
        Default is 1e-10.
    weno_order : UnsignedInteger, optional
        Order of the WENO scheme. Valid values are:
        1: Standard upwind scheme (order 1)
        2: WENO 2 (order 3)
        3: WENO 3 (order 5)
        Default is 3.

    See Also
    --------
    Structure

    """

    boundary_model = UnsignedInteger(default=0, ub=3)
    weno_eps = UnsignedFloat(default=1e-10)
    weno_order = UnsignedInteger(default=3, ub=3)
    _parameters = ['boundary_model', 'weno_eps', 'weno_order']


class ConsistencySolverParameters(Structure):
    """A class for defining the consistency solver parameters for Cadet.

    Parameters
    ----------
    solver_name : Switch, optional
        Name of the solver.
        Valid values are 'LEVMAR', 'ATRN_RES', 'ATRN_ERR', and 'COMPOSITE'.
        The default is 'LEVMAR'
    init_damping : UnsignedFloat, optional
        The initial damping parameter. Default is 0.01.
    min_damping : UnsignedFloat, optional
        The minimum damping parameter. Default is 0.0001.
    max_iterations : UnsignedFloat, optional
        The maximum number of iterations. Default is 50.
    subsolvers : Switch, optional
        Vector with names of solvers for the composite solver
        (only required for composite solver).
        Valid values are 'LEVMAR', 'ATRN_RES', 'ATRN_ERR', and 'COMPOSITE'.
        The default is 'LEVMAR'

    See Also
    --------
    Structure

    """

    solver_name = Switch(
        default='LEVMAR',
        valid=['LEVMAR', 'ATRN_RES', 'ARTN_ERR', 'COMPOSITE']
    )
    init_damping = UnsignedFloat(default=0.01)
    min_damping = UnsignedFloat(default=0.0001)
    max_iterations = UnsignedInteger(default=50)
    subsolvers = Switch(
        default='LEVMAR',
        valid=['LEVMAR', 'ATRN_RES', 'ARTN_ERR']
    )

    _parameters = [
        'solver_name', 'init_damping', 'min_damping',
        'max_iterations', 'subsolvers'
    ]
