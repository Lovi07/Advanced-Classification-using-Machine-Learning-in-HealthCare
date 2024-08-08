import pkg_resources

# List all installed packages and their versions
installed_packages = pkg_resources.working_set

# Create a list of package names and versions
package_versions = [(pkg.key, pkg.version) for pkg in installed_packages]

# Print package names and versions
for package_name, version in sorted(package_versions):
    print(f"{package_name}=={version}")
