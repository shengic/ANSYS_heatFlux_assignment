

Public Sub readAnsysFile()
   Dim stringLine As String, stringLine1 As String
   Dim regex As Object
   Set regex = CreateObject("VBScript.RegExp")
   Dim i As Integer, preLine As Integer, lineNumber As Integer
   Dim withinSection As Boolean
   Dim isNode As Boolean
   Dim isElement As Boolean
   Dim isHeatFlux As Boolean
   Dim inode As Integer
   Dim nextSection As String
   Dim totalLines As Long
   Dim ratio As Double
   Dim percentage As String
   Dim barWidth As Double
   
   Dim ansysNode As ansysCoordinate
   Dim ansysElementConnect As ansysElement
   Dim ansysHeatFluxConnect As ansysHeatFlux
   
  ' ansysFileName = "U:\private_shengic\ALS-U\ID\20221201 power implementation using excel\test1.dat"
  ' ansysFileName = "U:\private_shengic\ALS-U\ID\20221201 power implementation using excel\test2.dat"
   ansysFileName = form1.ansysFileName
   preLine = 0
   nextSection = "nothing"
   Debug.Print ("-------------------entering readAnsysFile--------")
   isNode = False
   isElement = False
   isHeatFlux = False
   withinSection = False
   
   ' estimate total line of the file
   totalLines = estimatedLineCount(ansysFileName)
   
   Open ansysFileName For Input As #1
   
'        Dim totalLines As Long
'        totalLines = WorksheetFunction.CountA(ActiveCell.EntireColumn)
        Dim currentLine As Long
        currentLine = 1
   
   Set ansysNodes = New Collection
   Set ansysElements = New Collection
   Set ansysHeatFluxElements = New Collection
   
   'Set ansysNode = New ansysCoordinate
   totalElementCount = 0

' show progress bar
   ratio = 0#
   progress ratio
   
   Do Until EOF(1)
     Line Input #1, stringLine
     
' estimate the ratio for status progress bar
         If currentLine Mod 1000 = 0 Then
           ratio = currentLine / totalLines
'Update the status bar
        'Application.StatusBar = "Reading ANSYS APDL file... " & Format(currentLine / totalLines, "0%") & " complete"

            If ratio > 1# Then
             ratio = 1#
            End If
            progress ratio
         End If
        
        'Process the current line here
        currentLine = currentLine + 1

   Select Case nextSection
'---------------------------------------node------------------------------------
    Case "node"
' start counting 3 lines before reading node info
         Select Case preLine
         Case Is < 3
           preLine = preLine + 1
           GoTo nextLine
         Case 3
' it is the third line and start to read, until the last line with string "-1"  and stop reading node info
           If stringLine = "-1" Then
            nextSection = "nothing"
            Debug.Print ("total nodes are " & ansysNodes.Count)
             GoTo nextLine
           Else
' third line within "node". start to record node info
            Set ansysNode = parseAnsysNodeLine(stringLine)
            ansysNodes.Add ansysNode, CStr(ansysNode.inode)
            preLine = 3
           GoTo nextLine
           End If
          End Select  ' end case preline
'---------------------------------------element------------------------------------
   Case "element"
' starting counting 4 lines before reading the element info
          Select Case preLine
          Case Is < 4
           preLine = preLine + 1
           GoTo nextLine
          Case 4
' it is the forth line and start to read, until the last line with string "-1" stop reading element info
           If stringLine = "-1" Then
            nextSection = "nothing"
            Debug.Print ("total element are " & ansysElements.Count)
            GoTo nextLine
           Else
' third element within "element". start to record element info
' read element info the first line
            If lineNumber = 1 Then
              stringLine1 = stringLine
              lineNumber = 2
            Else
'read element info the second line and add them together
              stringLine1 = stringLine1 & " " & stringLine
' dont store element info if the data is huge to avoid overflow
              'Set ansysElementConnect = parseAnsysElementLine(stringLine1)
              'ansysElements.Add ansysElementConnect, CStr(ansysElementConnect.element)
' since we are not going to store element info but we can count how many element
              totalElementCount = totalElementCount + 1
              lineNumber = 1
              GoTo nextLine
            End If
           End If
          End Select ' end case preline
'---------------------------------------heatfulx------------------------------------
    Case "heat flux"
' start counting 3 lines before reading node info
         Select Case preLine
         Case Is < 4
           preLine = preLine + 1
           GoTo nextLine
         Case 4
' it is the third line and start to read, until the last line with string "-1"  and stop reading node info
           If stringLine = "-1" Then
            nextSection = "nothing"
             Debug.Print ("total heat flux are " & ansysHeatFluxElements.Count)
             GoTo nextLine
           Else
' third line within "node". start to record node info
            Set ansysHeatFluxConnect = parseAnsysHeatFluxLine(stringLine)
            ansysHeatFluxElements.Add ansysHeatFluxConnect, CStr(ansysHeatFluxConnect.element)
           GoTo nextLine
           End If
          End Select  ' end case preline
'--------------------------------case else -------------------------------
    Case Else
'--------------------------------enter node info area---------------------
' check the leading strings if it is the beginning of the node info
    regex.Pattern = "Nodes for the whole assembly"
     If regex.Test(stringLine) Then
      preLine = 1
      nextSection = "node"
      GoTo nextLine
     End If
'--------------------------------enter element info area-------------------
' check the leading strings if it is the beginning of the element info
    regex.Pattern = "/com,\*{1,} Elements for"
     If regex.Test(stringLine) Then
      preLine = 1
      lineNumber = 1
      nextSection = "element"
      GoTo nextLine
      End If
'--------------------------------enter heat flux info area-----------------
' check the leading strings if it is the beginning of the heat flux info
    regex.Pattern = "/com,\*{1,} Create ""Heat Flux"""
     If regex.Test(stringLine) Then
      preLine = 1
      nextSection = "heat flux"
      GoTo nextLine
     End If
' loop to the next line for catch all
     GoTo nextLine

    End Select ' end nextSection case
nextLine:
   Loop
   Close #1

     ratio = 1#
     progress ratio
     
   Debug.Print ("-----------------leaving readAnsysFile------------")
End Sub



Public Function parseAnsysNodeLine(stringLine As String) As ansysCoordinate
   Dim stringArray() As String
   stringLine = removeAdditionalSpaceForSplit(stringLine)
   stringArray() = Split(stringLine)
   Set parseAnsysNodeLine = New ansysCoordinate
   parseAnsysNodeLine.inode = CLng(stringArray(0))
   parseAnsysNodeLine.x = CDbl(stringArray(1))
   parseAnsysNodeLine.y = CDbl(stringArray(2))
   parseAnsysNodeLine.z = CDbl(stringArray(3))
   
End Function

Public Function parseAnsysElementLine(stringLine As String) As ansysElement
   Dim stringArray() As String, i As Long
   Dim numbers As Variant
   Dim number As Variant
   Dim nodes As Collection
' element number start from 11-th to 30-th array for quadralaterial element
'   numbers = Array(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
' element number start from 11-th to 30-th array for trapezoidal element
   numbers = Array(11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
   
   'Dim number As Integer
   stringLine = removeAdditionalSpaceForSplit(stringLine)
   stringArray() = Split(stringLine)
   Set parseAnsysElementLine = New ansysElement
' 10th array is the element number
   parseAnsysElementLine.element = CInt(stringArray(10))
   i = 0
' from 11- 30-th are nodal number for that element
    Set nodes = New Collection
   For Each number In numbers
     i = i + 1
     nodes.Add CLng(stringArray(number)), CStr(i)
   Next
   Set parseAnsysElementLine.elementNodes = nodes
End Function


Public Function parseAnsysHeatFluxLine(stringLine As String) As ansysHeatFlux
   Dim stringArray() As String
   Dim numbers As Variant
   Dim number As Variant
   Dim nodes As Collection
   Dim i As Long
 ' element number start from 11-th to 30-th array
 ' 5-12 are nodal of the heat flux surface nodes, 5,6,7,8 are corner nodes, 9,10,11,12 are midpoint nodes
   numbers = Array(5, 6, 7, 8, 9, 10, 11, 12)
   stringLine = removeAdditionalSpaceForSplit(stringLine)
   stringArray() = Split(stringLine)
' 0-th element is the surface element (usually is the successive number of the element number
'   parseAnsysHeatFluxLine.element = CInt(stringArray(0))
   Set parseAnsysHeatFluxLine = New ansysHeatFlux
   i = 0
' first item in the ANSYS file is the element number
   parseAnsysHeatFluxLine.element = stringArray(0)
   Set nodes = New Collection
   For Each number In numbers
     i = i + 1
    ' nodes.Add CInt(stringArray(number)), CStr(i)
    nodes.Add ansysNodes.Item(stringArray(number)), CStr(i)
   Next
    Set parseAnsysHeatFluxLine.elementNodes = nodes
' calculate the heat flux surface area
    Call parseAnsysHeatFluxLine.calculateHeatFluxSurfaceArea
End Function

Public Function removeAdditionalSpaceForSplit(stringLine As String) As String
   Dim regex As Object
   Set regex = CreateObject("VBScript.RegExp")
   regex.Global = True
   regex.Pattern = "^\s+"
   stringLine = regex.Replace(stringLine, "")
   regex.Pattern = "\s+"
   stringLine = regex.Replace(stringLine, " ")
   removeAdditionalSpaceForSplit = stringLine
End Function

Private Sub progress(ratio As Double)

form1.ansysFileProgressDescription.Caption = Format(ratio, "0%") & " read"
form1.ansysFileProgressBar.Width = ratio * 160
DoEvents

End Sub