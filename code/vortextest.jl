# Load the module and generate the functions
module CxxVortexTest
  using CxxWrap
  @wrapmodule("build/vortextest_jlcxx")

  function __init__()
    @initcxx
  end
end
