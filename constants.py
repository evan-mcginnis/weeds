#
# C O N S T A N T S
#
# Various constants

# Names for the attributes
NAME_AREA = "area"
NAME_TYPE = "type"
NAME_LOCATION = "location"
NAME_CENTER = "center"
NAME_CONTOUR = "contour"
NAME_SHAPE_INDEX = "shape_index"
NAME_RATIO = "lw_ratio"
NAME_IMAGE = "image"
NAME_REASON = "reason"
NAME_SIZE_RATIO = "size_ratio"
NAME_BLUE = "blue"
NAME_DISTANCE = "distance"
NAME_NAME = "name"

# A shortcut used for the command line
NAME_ALL = "all"

names = [NAME_AREA, NAME_TYPE, NAME_LOCATION, NAME_CENTER, NAME_CONTOUR, NAME_SHAPE_INDEX, NAME_RATIO, NAME_IMAGE, NAME_REASON]

# REASONS why things were scored the way they were
REASON_UNKNOWN = 0
REASON_AT_EDGE = 1
REASON_SIZE_RATIO = 2
REASON_LOGISTIC_REGRESSION = 3
REASONS = ["Unknown", "At Edge", "Size", "Logistic"]

# TYPES of vegetation
TYPE_DESIRED = 0
TYPE_UNDESIRED = 1
TYPE_UNTREATED = 2
TYPE_IGNORED = 3
TYPE_UNKNOWN = 4
TYPES = ["Desired", "Undesired", "Untreated", "Ignored", "Unknown"]

# Properties
PROPERTY_PIXELS_PER_MM = "PIXELS-PER-MM"
