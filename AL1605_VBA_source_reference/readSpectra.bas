Option Explicit


Public Sub readSpectraFile()
   Dim strLine As String
   Dim stringArray() As String
   'ReDim powerArray(1 To 100)
   'Dim powerList As arrayList
   Dim i As Long
'   Public spectraTotalNode As Long
'   Public spectraTotalColumn As Long
'   Public spectraTotalRow As Long
   Dim inode As Long
   Dim totalNode As Long
   Dim previousYvalue As String
   Dim spectraPowerElement As powerDensityElement
   Dim ygrids() As Variant
   Dim igrid As Integer
   Dim currentLine As Long
   Dim totalLines As Long
   Dim ratio As Double
   Dim totalPowerDensity As Double
   
   Set SpectraPowerNodes = New Collection
   
   'Set powerList = New arrayList
   Debug.Print ("-------------------entering readSpectraFile--------")
   'spectraFileName = "U:\private_shengic\ALS-U\ID\20221201 power implementation using excel\Tender power.dta"
   'spectraFileName = "U:\private_shengic\ALS-U\ID\20221201 power implementation using excel\spectratest.txt"
   spectraFileName = form1.spectraFileName
   i = 0
   inode = 0
   spectraTotalColumn = 0
   spectraTotalRow = 0
   previousYvalue = ""
   igrid = 0
   
   spectraTotalPower = 0#
   spectraPeakPowerDensity = 0#
   totalPowerDensity = 0#
   spectraGridArea = 0#
   
  ' estimate total line of the file
   totalLines = estimatedLineCount(spectraFileName)
   
   Open spectraFileName For Input As #1
   Do Until EOF(1)
      Line Input #1, strLine
      
    ' Check if the line is empty, exit the loop if true
    If Trim(strLine) = "" Then Exit Do
   
   ' estimate the ratio for status progress bar
         If currentLine Mod 1000 = 0 Then
           ratio = currentLine / (2 * totalLines)
            If ratio > 1# Then
             ratio = 0.5
            End If
            progress ratio
         End If
        
        'Process the current line here
        currentLine = currentLine + 1
   
   stringArray() = Split(parseRawSpectraLine(strLine))
    If (i > 1) Then
        'Dim dict
        'Set dict = CreateObject("Scripting.Dictionary")
        Dim powerDNode As powerDensityNode
        Set powerDNode = New powerDensityNode
        inode = inode + 1
' read power density and put the nodal and power info to the collection
        Call powerDNode.initialized(inode, CDbl(stringArray(0)), _
                                           CDbl(stringArray(1)), _
                                           CDbl(stringArray(2)))
        SpectraPowerNodes.Add powerDNode, CStr(inode)
'totalPowerDensity is checking only
        totalPowerDensity = totalPowerDensity + powerDNode.powerDensity
        If spectraPeakPowerDensity <= CDbl(stringArray(2)) Then
           spectraPeakPowerDensity = CDbl(stringArray(2))
        End If

' check if current y value is the same as previous value, if yes then that is the end of x coloum value (total column number)
       If stringArray(1) <> previousYvalue And spectraTotalColumn = 0 And previousYvalue <> "" Then
         spectraTotalColumn = inode - 1
       Else
         previousYvalue = stringArray(1)
       End If
    End If
     i = i + 1
   Loop
   
   Close #1


   spectraTotalNode = SpectraPowerNodes.Count
   spectraTotalRow = spectraTotalNode / spectraTotalColumn
   Debug.Print ("total Spectra power node size =" & spectraTotalNode)
   Debug.Print ("total row and column =" & spectraTotalRow & ", " & spectraTotalColumn)

   ' generate spectra element connectivity
   Dim elementNodes As Collection
   Dim spectraElement As Long
   Dim spectraTotalElement As Long
   Dim powerDElement As powerDensityElement
   Set spectraPowerElements = New Collection
   
' initialize spectraTotalPower
    spectraTotalPower = 0#
    
    Dim maxPowerDensity As Double
    Dim tempTotalPowerDensity As Double
    Dim tempTotalPower As Double
    maxPowerDensity = 0#
    tempTotalPowerDensity = 0#
    tempTotalPower = 0#
    
    spectraTotalElement = (spectraTotalRow - 1) * (spectraTotalColumn - 1)
    
' form all 4 nodes power element collection --------------------------------------
    For spectraElement = 1 To spectraTotalElement
    If spectraElement Mod 1000 = 0 Then

      ratio = 0.5 + spectraElement / (2# * spectraTotalElement)
      If ratio > 1# Then
         ratio = 1#
      End If
    End If
    Set powerDElement = New powerDensityElement
     Call powerDElement.initialized(spectraElement, spectraTotalColumn)
     spectraPowerElements.Add powerDElement, CStr(spectraElement)
    ' spectraTotalPower = spectraTotalPower + powerDElement.totalPower
    ' pick up area only once, because rest of them are the same
     If spectraElement = 1 Then
      spectraGridArea = powerDElement.elementArea
    ' since i have totalPowerDensity just multiple area we should have total power
    '  spectraTotalPower = totalPowerDensity * spectraGridArea
     End If
    
     If maxPowerDensity <= powerDElement.averagePowerDensity Then
       maxPowerDensity = powerDElement.averagePowerDensity
     End If
       tempTotalPowerDensity = tempTotalPowerDensity + powerDElement.averagePowerDensity
  '     tempTotalPower = tempTotalPower + powerDElement.totalPower
     spectraTotalPower = spectraTotalPower + powerDElement.totalPower
    Next spectraElement
    
'maxPowerDensity is the max after taking the max. spectraPeakPowerDensity is strictly compared from raw data
' so i would overwrite it
    spectraPeakPowerDensity = maxPowerDensity
    
'--------------------------------------------------------------------------------
   ' move progress bar to 100%
     ratio = 1#
     progress ratio

   Debug.Print ("spectra total element = " & (spectraTotalColumn - 1) & " x " & (spectraTotalRow - 1) & " = " & spectraTotalElement)
   Debug.Print ("-------------------leaving readSpectraFile-------------------")
End Sub


Public Function parseRawSpectraLine(stringLine As String) As String
   Dim regex As Object
   'Set regEx = New RegExp
   Set regex = CreateObject("VBScript.RegExp")
   regex.Global = True
   'Pattern = "(/)|(,)|(-)"
   'Pattern = ","
   regex.Pattern = "^\s+"
   stringLine = regex.Replace(stringLine, "")
   regex.Pattern = "\s+"
   stringLine = regex.Replace(stringLine, " ")
   parseRawSpectraLine = stringLine
End Function


Private Sub progress(ratio As Double)

form1.spectraFileProgressDescriptioin.Caption = Format(ratio, "0%") & " read"
form1.spectraFileProgressBar.Width = ratio * 160
DoEvents

End Sub