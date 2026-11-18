# Coupled prescribed axial sources

Contract version 2 ideal velocity sources may use `motion_profile: rigid_translation`
with a finite, nonzero three-component `motion_axis`. The compiled driver normalizes
this direction and enforces active physical symmetry planes. The scalar excitation
is axial velocity, multiplied by positive per-boundary motion weights.

A component may own FEM boundaries, BEM boundaries, or both. BEM forcing uses the
signed dot product of the unit axis and each BEM face normal. FEM forcing uses the
normal pointing outward from the owning tetrahedral acoustic volume, independent
of triangle winding. The same projection multiplies the existing Neumann load in
both monolithic and condensed solvers, including affine P2 boundary integration.
FEM and BEM forcing retain independent complex excitation bases and the existing
phasor convention. Legacy surface-normal requests retain their existing loads.

Workers advertise `coupled_source_profiles` for sources with FEM assignments.
The older `coupled_exterior_source_profiles` capability remains sufficient for
exterior-only assignments in coupled models. Clients refuse axial FEM assignments
when the new capability is missing. Interior-only prescribed axial sources remain
unsupported. No schema or protocol version changes are required.

Validation covers opposite/tangential axes, winding-independent FEM outward
normals, P2 loads, complex monolithic/condensed agreement and reversal, worker
capability negotiation, and application-to-worker coupled source solves.
