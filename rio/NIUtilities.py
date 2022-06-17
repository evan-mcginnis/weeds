#
# N A T I O N A L  I N S T R U M E N T S  U T I L I T I E S
#
# Evan McGinnis


# Pieces for how NI names their ports
NI_PORT = "port"
NI_PORT0 = "port0"
NI_LINE = "line"
NI_SEPARATOR = "/"

def channelName(module: str, port: int, line: int) -> str:
    return module + NI_SEPARATOR + NI_PORT + str(port) + NI_SEPARATOR + NI_LINE + str(line)

def breakdown(line: str) -> []:
    return line.split(NI_SEPARATOR)

