

# Load the module and generate the functions
module CxxVortexTest

    module_path, _ = splitdir(@__FILE__);   # Path to this module

    using CxxWrap
        @wrapmodule(joinpath(module_path, "build/vortextest_jlcxx"))

        function __init__()
        @initcxx
    end
end
