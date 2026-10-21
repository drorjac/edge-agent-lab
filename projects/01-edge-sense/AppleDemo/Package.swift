// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "ActivityDemo",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "activity-demo", targets: ["ActivityDemo"])],
    targets: [.executableTarget(name: "ActivityDemo")]
)
