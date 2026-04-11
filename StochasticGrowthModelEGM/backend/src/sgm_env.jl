using Pkg

project_dir = @__DIR__
Pkg.activate(project_dir)
Pkg.instantiate()               # installs what’s in Project.toml/Manifest.toml

println("Environment ready at: ", project_dir)

include(joinpath(project_dir, "sgm_main.jl"))

