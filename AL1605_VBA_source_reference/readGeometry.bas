Public Sub readGeometry(geoFile As String)
 Dim InputFile As String
 Dim InputLine As String
 Dim InputData() As String
 Dim i As Integer

' Set the input file name
 InputFile = geoFile

' Open the input file
Open InputFile For Input As #1

' Read the input file line by line
Do Until EOF(1)
    InputLine = Input(1, #1)
    ' Check if the line starts with "#", Trim function is to remove spaces on both ends of a string
    If Left(Trim(InputLine), 1) <> "#" Then
        ' Split the line into an array using the comma delimiter
        InputData = Split(InputLine, ",")
        ' the ending character can be : or =
         Select Case splitString(InputData(0), "[:=]+")
         
         Case "spectra filename"
           spectraFileName = InputData(1)
         Case "ansys filename"
           ansysFileName = InputData(1)
         Case "source"
           xSource = InputData(1)
           ySource = InputData(2)
           zSource = InputData(3)
         Case "target"
           xTarget = InputData(1)
           yTarget = InputData(2)
           zTarget = InputData(3)
         Case "side"
           xSide = InputData(1)
           ySide = InputData(2)
           zSide = InputData(3)
         End Select
        ' Process the input data
        'For i = 0 To UBound(InputData)
        '    Debug.Print InputData(i)
        Next i
    End If
Loop

' Close the input file
Close #1
End Sub


Function splitString(text As String, delimiter As String) As Variant
Dim regex As Object
Set regex = CreateObject("VBScript.RegExp")
regex.Global = True
regex.Pattern = delimiter
splitString = regex.Execute(text)
End Function