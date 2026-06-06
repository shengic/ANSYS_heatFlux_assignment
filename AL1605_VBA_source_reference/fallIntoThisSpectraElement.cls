   Dim xgrids(4000) As Double
   Dim ygrids(4000) As Double

Public Sub initialize()
  Call determineGrid
End Sub


Sub determineGrid()
   Dim xelement() As Long
   Dim yelement() As Long
   Dim ixelement As Long
   Dim iyelemenet As Long
   Dim nextiyelement As Long
   Dim powerElement As powerDensityElement
   Dim xrange As Object
   Dim fourNodes As Collection

   
   minX = 1000#
   maxX = -1000#
   minY = 1000#
   maxY = -1000#
   
'      (x1,y2)
'       1-------3
'       |       |
'       |       |
'       |       |
'       2-------4
'              (x2,y1)
   
' store y value to be used for later algorithm to determine if any point fall into specific spectra
 '  ReDim xgrids(spectraTotalColumn)
   ReDim xelement(spectraTotalColumn - 1)
 '  ReDim ygrids(spectraTotalRow)
   ReDim yelement(spectraTotalRow - 1)
' obtain top first row element then their corresponding upper left coordinate

   For ixelement = 1 To spectraTotalColumn - 1
      xelement(ixelement) = ixelement
      Set powerElement = spectraPowerElements.Item(CStr(xelement(ixelement)))

      xgrids(ixelement) = powerElement.x1
      xgrids(ixelement + 1) = powerElement.x2
      
'      xgrids(ixelement) = SpectraPowerNodes(fourNodes.Item(1)).xmrad
'      xgrids(ixelement + 1) = SpectraPowerNodes(powerElement(3)).xmrad
      'Debug.Print ("ixelement = " & ixelement)
   Next ixelement
   
   minX = xgrids(1)
   maxX = xgrids(spectraTotalColumn)
   
   For iyelement = 1 To spectraTotalRow - 1
      yelement(iyelement) = (iyelement - 1) * (spectraTotalColumn - 1) + 1
      'nextiyelement =  (iyelement ) * (spectraTotalColumn - 1) + 1
      Set powerElement = spectraPowerElements.Item(CStr(yelement(iyelement)))
      'Debug.Print ("element 1 and 2 = " & SpectraPowerNodes(powerElement(1)) & ", " & SpectraPowerNodes(powerElement(2)))
       ygrids(iyelement) = powerElement.y2
       ygrids(iyelement + 1) = powerElement.y1
'      ygrids(iyelement) = SpectraPowerNodes(powerElement(1)).ymrad
'      ygrids(iyelement + 1) = SpectraPowerNodes(powerElement(2)).ymrad
      'Debug.Print ("iyelement = " & yelement(iyelement))
   Next iyelement
   
   minY = ygrids(1)
   maxY = ygrids(spectraTotalRow)
   
   Debug.Print ("done with determine Grid")
End Sub


Public Function fallIntoThisSpectraElement(xmrad As Double, ymrad As Double) As Long
  Dim i As Long
  Dim j As Long
  
  fallIntoThisSpectraElement = 0
  
' if provide points is inside of the spectra range, give no element number
  If (xmrad >= minX And xmrad <= maxX) And (ymrad >= minY And ymrad <= maxY) Then
' check along column
     For i = 1 To spectraTotalColumn - 1
       If xmrad >= xgrids(i) And xmrad <= xgrids(i + 1) Then
' if it falls into the column then check row
         For j = 1 To spectraTotalRow - 1
            If ymrad >= ygrids(j) And ymrad <= ygrids(j + 1) Then
              fallIntoThisSpectraElement = (j - 1) * (spectraTotalColumn - 1) + i
            GoTo nextLine
            End If
         Next j
       End If
     Next i
  End If
nextLine:
End Function

Public Function getThisSpectraElementNodes(whichElement As Long) As Collection
 Dim i As Long
 Dim inode As Long
 Dim powerOneNode As powerDensityNode
 Dim nodes As Collection
 Set nodes = New Collection
 Set getThisSpectraElementNodes = New Collection
 Set spectraOneElement = spectraElementConnectivity.Item(whichElement)
 For i = 1 To 4
  inode = spectraOneElement.Item(i)
     nodes.Add SpectraPowerNodes.Item(inode), CStr(i)
    Set powerOneNode = SpectraPowerNodes.Item(inode)
    Debug.Print ("node " & i & "=" & inode & _
    " xmrad=" & powerOneNode.xmrad & " ymrad=" & powerOneNode.ymrad)
 Next i
End Function