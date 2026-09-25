// openapi-typescript drives the TypeScript compiler API at runtime, which
// TypeScript 7 (the native Go port) no longer ships. Its `typescript` peer
// would otherwise resolve to the project's 7.x, so give it a private copy of
// the last 5.x line its peer range asks for. The project type-checks with 7.x.
function readPackage(pkg) {
  if (pkg.name === "openapi-typescript") {
    const { typescript: _peer, ...peers } = pkg.peerDependencies ?? {};
    pkg.peerDependencies = peers;
    pkg.dependencies = { ...pkg.dependencies, typescript: "5.9.3" };
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
