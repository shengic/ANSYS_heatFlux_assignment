Option Explicit

' spectra file name
   Global spectraFileName As String
' ansys file name
   Global ansysFileName As String
' ansysNodes is a collection to store all ansys nodes number and x,y,z coordinates
   Global ansysNodes As Collection
' ansysElements is a collection to store all ansys element number and connectivity
   Global ansysElements  As Collection
   Global interpolateAnsysHeatFlux As Collection
   Global totalElementCount As Long
' external data file path
   Global externalFilePath As String
'  total power ratio
   Global totalPowerRatio As Double
' source geometry
   Global xSource As Double
   Global ySource As Double
   Global zSource As Double
   Global xTarget, yTarget, zTarget As Double
   Global xSide, ySide, zSide As Double

  ' Dim xV() As Double
  ' Dim yV() As Double
  ' Dim zV() As Double
   Global geometry As sourceGeometry
'----------------------------------------------------------------------------------
' ansysHeatFLuxElements is a collection all heatflux elements
' each element is a module "ansysHeatFlux"
' which has "element" as the element number (doesnt really useful)
' "elementNodes" which stores 8 nodes, first 4 nodes are the corner nodes
   Global ansysHeatFluxElements As Collection
'----------------------------------------------------------------------------------
' SpectraPowerNodes is a collection store spectra node info, each element is a module
' "powerDensityNode"
' which contains inode, xmrad, ymrad and powerDensity
   Public SpectraPowerNodes As Collection
'----------------------------------------------------------------------------------
' spectraPowerElements is a collection store all spectra element
' each element is a module powerDensityElement item 1-4th are fourNodes collection, each
' collection is powerDensityNode module
'       1-------3
'       |       |
'       |       |
'       |       |
'       2-------4
   Public spectraPowerElements As Collection
'----------------------------------------------------------------------------------

' use in calculateAnsysPowerDensity
   Public minX As Double
   Public maxX As Double
   Public minY As Double
   Public maxY As Double

   Public spectraTotalNode As Long
   Public spectraTotalElement As Long
   Public spectraTotalColumn As Long
   Public spectraTotalRow As Long
   Public spectraPeakPowerDensity As Double
   Public spectraTotalPower As Double
   Public spectraGridArea As Double  ' each grid area

Public Sub showForm()
  form1.Show
End Sub

Public Sub readAll()
' read ansys file and store the data
   Call readAnsysFile
' read the spectra file and store the data
   Call readSpectraFile
   
   Debug.Print ("size of spectraPowerElements =" & spectraPowerElements.Count)
   Debug.Print ("done with reading ansys and spectra files")
' determine the centroid point of a heatflux surface to ansysHFluxElement.elementNodes
   Call storeAnsysInterpolateCoordinateAndPower
' write out to a external file
   Call writeAnsysHeatFluxElements
  Debug.Print ("finished all so far")
End Sub

Public Sub writeAnsysHeatFluxElements()
  Dim fileName As String
  Dim FilePath As String
  Dim outputLine As String
  Dim i As Integer
  Dim heatFlux As ansysHeatFlux
  Dim arrayString(10) As String
  Dim totalPower As Double 'total power calculated for the entire heat flux
  Dim currentLine As Long
  Dim ratio As Double
  
  currentLine = 1
  totalPower = 0#
  externalFilePath = ""


'  fileName = "output.txt"
'  filePath = "U:\private_shengic\ALS-U\ID\20221201 power implementation using excel\"
  FilePath = form1.outputFilePath.Value
  fileName = form1.outputFileName.Value
  form1.outputTotalElement.Caption = Format(ansysHeatFluxElements.Count, "#,##0")
  form1.totalPowerFromSpectra.Caption = form1.spectraTotalPower1.Caption
  externalFilePath = FilePath & "\" & fileName

' start to write out external file -------------------------------------------
  ratio = 0#
  progress ratio
  totalPowerRatio = form1.totalPowerRatio.text
  If totalPowerRatio = 0 Then
   totalPowerRatio = 1#
  End If

  Open externalFilePath For Output As #1
   For i = 1 To ansysHeatFluxElements.Count
   Set heatFlux = ansysHeatFluxElements.Item(i)
' take discount of the current total power by totalPowerRatio
   totalPower = totalPower + heatFlux.totalPower_W * totalPowerRatio
' write out x,y,z and projected power density
   arrayString(0) = heatFlux.x
   arrayString(1) = heatFlux.y
   arrayString(2) = heatFlux.z
   arrayString(3) = heatFlux.projectedPowerDensityIn_W_mm2 * totalPowerRatio
   arrayString(4) = heatFlux.projectedPowerDensityIn_W_mm2
   'arrayString(5) = heatFlux.grazingAngleDegree
  ' arrayString(6) = heatFlux.surfaceArea_mm2
   
   outputLine = ArrayToCommaDelimitedString(arrayString)
   Print #1, outputLine
   
' estimate the ratio for status progress bar
         If currentLine Mod 1000 = 0 Then
           ratio = currentLine / ansysHeatFluxElements.Count
'Update the status bar
        'Application.StatusBar = "Reading ANSYS APDL file... " & Format(currentLine / totalLines, "0%") & " complete"

            If ratio > 1# Then
             ratio = 1#
            End If
            progress ratio
         End If
        
        'Process the current line here
        currentLine = currentLine + 1
   Next i
  Close #1
  
   ratio = 1#
   progress ratio
   
  Debug.Print ("Write out file output.txt")
  form1.viewFileButton.Enabled = True
  form1.outputTotalPower.Caption = Format(totalPower / 1000#, "0.00") & " kW"
  'form1.totalPowerRatio.text = Format(totalPower / (spectraTotalPower * 1000#), "Percent")
  'totalPowerRatio = totalPower / (spectraTotalPower * 1000#)
  'form1.reCalculateTotalPowerButton.Enabled = True
  MsgBox "External file " & form1.outputFileName.Value & " for ANSYS has been created."
End Sub

' this is the sub to determine if the geometry points falls into the spectra grid and store interpolate power

Public Sub storeAnsysInterpolateCoordinateAndPower()
   Dim i, j, k As Long
   Dim xmrad As Double
   Dim ymrad As Double
   Dim nodes As Collection
   Dim node As ansysCoordinate
   Dim x, y, z As Double
   Dim whichElement As Long
   Dim ansysHFluxElement As ansysHeatFlux
   Dim thisPowerElement As powerDensityElement

' prepare  fallIntoThisSpectraElement
   Dim thisSpectraElement As fallIntoThisSpectraElement
   Set thisSpectraElement = New fallIntoThisSpectraElement
   Call thisSpectraElement.initialize
   
' for testing if it fall into the spectra element only
'   xmrad = -1#
'   ymrad = 1.92
'   whichElement = thisSpectraElement.fallIntoThisSpectraElement(xmrad, ymrad)
'  Set spectraOneElement = calculateAnsysPower.getThisSpectraElementNodes(whichElement)
'   Set thisPowerElement = spectraPowerElements.Item(CStr(whichElement))
'   Debug.Print ("fall into the element = " & whichElement)
   
   
' setup geometry from the CAD model related to the source point
   Set geometry = New sourceGeometry
   Call geometry.initialized
   
'   Set interpolateAnsysHeatFlux = New Collection

   For i = 1 To ansysHeatFluxElements.Count
        Set ansysHFluxElement = ansysHeatFluxElements.Item(i)
        x = 0#: y = 0#: z = 0#
'find average coordinate in a 4 node heating surface
         For j = 1 To 4
          x = x + ansysHFluxElement.elementNodes.Item(j).x / 4#
          y = y + ansysHFluxElement.elementNodes.Item(j).y / 4#
          z = z + ansysHFluxElement.elementNodes.Item(j).z / 4#
         Next j
' store it to ansysHeatFlux x,y,z double
       '  Call ansysHFluxElement.setInterpolateCoordinate(x, y, z)
         ansysHFluxElement.x = x
         ansysHFluxElement.y = y
         ansysHFluxElement.z = z
' map this coordinate to mrad/mrad coordindate based on geometry coordinate in the model
         Call ansysHFluxElement.mapAnsysCoordinateToMrad
         whichElement = thisSpectraElement.fallIntoThisSpectraElement( _
                        ansysHFluxElement.xmrad, ansysHFluxElement.ymrad)
         If whichElement > 0 Then
           Set thisPowerElement = spectraPowerElements.Item(CStr(whichElement))
           ansysHFluxElement.spectraElement = whichElement
           Call ansysHFluxElement.calculateInterpolatePower(thisPowerElement)
           'Debug.Print ("this Flux node fall into the spectraElenent " & whichElement)
         Else
           'Debug.Print ("this Flux node is not in the region for the spectraPower")
         End If
         
   Next i
End Sub


Public Function ArrayToCommaDelimitedString(inputArray As Variant) As String
    Dim outputString As String
    Dim i As Integer
    For i = LBound(inputArray) To UBound(inputArray)
        If inputArray(i) <> "" Then
            outputString = outputString & Format(inputArray(i), "0.0000E+00") & ",    "
        Else
            outputString = outputString & Format(inputArray(i), "0.0000E+00")
        End If
    Next i
    If Right(outputString, 1) = "," Then
        outputString = Left(outputString, Len(outputString) - 1)
    End If
    ArrayToCommaDelimitedString = outputString
End Function


Public Function PrintNumberWithDecimalPlaces(inputValue As Variant, decimalPlaces As Integer) As String
    Dim outputValue As Double
    'Try to convert the input value to a number
    On Error Resume Next
    outputValue = CDbl(inputValue)
    On Error GoTo 0
    
    'If the input value can't be converted to a number, return an error message
    If outputValue = 0 And Not IsNumeric(inputValue) Then
        PrintNumberWithDecimalPlaces = "Error: Input is not a number"
        Exit Function
    End If
    
    'Print the output value with the specified number of decimal places
    PrintNumberWithDecimalPlaces = Format(outputValue, "0." & String(decimalPlaces, "0"))
End Function


Public Function estimatedLineCount(fileName As String) As Long
    Dim fso As New FileSystemObject
    Dim file As TextStream
    Dim FilePath As String
    Dim fileSize As Long
    Dim approxLineCount As Long
    
    ' Set the path of the file to be estimated
    FilePath = fileName
    
    ' Get the file size in bytes
    fileSize = fso.GetFile(FilePath).Size
    
    ' Estimate the number of lines based on the file size
    approxLineCount = fileSize \ 50  ' assuming average line length of 80 characters
    
    ' Display the estimated line count
    'MsgBox "Estimated line count: " & approxLineCount
    estimatedLineCount = approxLineCount
End Function

Private Sub progress(ratio As Double)

form1.ansysHeatFluxFileProgressDescription.Caption = Format(ratio, "0%") & " output"
form1.ansysHeatFluxFileProgressBar.Width = ratio * 160
DoEvents

End Sub