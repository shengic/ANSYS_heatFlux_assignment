' calculate the ANSYS heat flux element power density information
Public element As Long
Public elementNodes As Collection
Public x As Double
Public y As Double
Public z As Double
Public vx As Double
Public vY As Double
Public vz As Double

Public xmrad As Double
Public ymrad As Double

Public distanceFromSource As Double
Public spectraElement As Long
Public normalPowerDensity As Double
Public projectedPowerDensity As Double
Public grazingAngleRadian As Double
Public grazingAngleDegree As Double
Public projectedPowerDensityIn_W_mm2 As Double
Public normalPowerDensityIn_W_mm2 As Double
Public surfaceArea_mm2 As Double ' surface area on this heatflux element
Public totalPower_W As Double ' total power on this heatflux surface


' convert ANSYS (x,y,z) coordindate into (mrad x, mrad y) coordindate
Public Sub mapAnsysCoordinateToMrad()
 Dim dotProductVx As Double
 Dim dotProductVy As Double
 Dim getVX() As Double
 Dim getVY() As Double
 Dim getVZ() As Double
 
 ReDim getVX(2), getVY(2), getVZ(2)
 
' transfer coordindate based on source point
  vx = x - xSource
  vY = y - ySource
  vz = z - zSource
  distanceFromSource = Sqr(vx ^ 2 + vY ^ 2 + vz ^ 2)
  
  getVX = geometry.getVX
  getVY = geometry.getVY
  getVZ = geometry.getVZ
  
 dotProductVx = vx * getVX(0) + vY * getVX(1) + vz * getVX(2)
 dotProductVy = vx * getVY(0) + vY * getVY(1) + vz * getVY(2)
  
  xmrad = WorksheetFunction.Asin(dotProductVx / distanceFromSource) * 1000#
  ymrad = WorksheetFunction.Asin(dotProductVy / distanceFromSource) * 1000#
  
' atan2(x,y) times 1000 to get mrad unit
'  xmrad = WorksheetFunction.Atan2(vz, vx) * 1000#
'  ymrad = WorksheetFunction.Atan2(vz, vy) * 1000#
End Sub

' determine the heat flux surface area
Public Sub calculateHeatFluxSurfaceArea()
' calculate the surface area surfaceArea_mm2
surfaceArea_mm2 = quadrilaterialArea(elementNodes.Item(1), _
                                     elementNodes.Item(2), _
                                     elementNodes.Item(3), _
                                     elementNodes.Item(4))
End Sub

'calculate interpolated power using given spectra four notes power density
Public Sub calculateInterpolatePower(spectraElement As powerDensityElement)
  Dim a As Collection
  
  Set a = spectraElement.interpolateCoefficients
  normalPowerDensity = a.Item(1) + _
                 a.Item(2) * xmrad + _
                 a.Item(3) * ymrad + _
                 a.Item(4) * xmrad * ymrad
grazingAngleRadian = getGrazingAngleInRadian()
' grazing angle
grazingAngleDegree = grazingAngleRadian * 180# / WorksheetFunction.Pi
' projected power densityin kW/mrad^2
projectedPowerDensity = Sin(grazingAngleRadian) * normalPowerDensity

'into W/mm^2
normalPowerDensityIn_W_mm2 = normalPowerDensity * 1000 / (distanceFromSource / 1000#) ^ 2
projectedPowerDensityIn_W_mm2 = Sin(grazingAngleRadian) * normalPowerDensityIn_W_mm2
' calculate the surface area surfaceArea_mm2
'surfaceArea_mm2 = quadrilaterialArea(elementNodes.Item(1), _
'                                     elementNodes.Item(2), _
'                                     elementNodes.Item(3), _
'                                     elementNodes.Item(4))
' calculate the total power of this element
totalPower_W = surfaceArea_mm2 * projectedPowerDensityIn_W_mm2

End Sub

'calculate the grazing angle between the beam ray coming to this element and the normal vector
' of this element surface
' j->i x j->k = outnormal vector

Public Function getGrazingAngleInRadian() As Double
 Dim vx1, vy1, vz1 As Double
 Dim vx2, vy2, vz2 As Double
 Dim nd, sd As Double
' normal vector
 Dim nx, ny, nz As Double
' source point vector is vx,vy and vz
 
'vector j->i
 vx1 = elementNodes.Item(1).x - elementNodes.Item(2).x
 vy1 = elementNodes.Item(1).y - elementNodes.Item(2).y
 vz1 = elementNodes.Item(1).z - elementNodes.Item(2).z

' vector j->k
 vx2 = elementNodes.Item(3).x - elementNodes.Item(2).x
 vy2 = elementNodes.Item(3).y - elementNodes.Item(2).y
 vz2 = elementNodes.Item(3).z - elementNodes.Item(2).z
 
' normal vector by cross product of vector 1 x vector 2 (ji x jk)
 nx = vx1 * vy2 - vy1 * vx2
 ny = vy1 * vz2 - vz1 * vy2
 nz = vz1 * vx2 - vx1 * vz2

 nx = (vy2 * vz1) - (vz2 * vy1)
 ny = (vz2 * vx1) - (vx2 * vz1)
 nz = (vx2 * vy1) - (vy2 * vx1)
 
 nd = Sqr(nx ^ 2 + ny ^ 2 + nz ^ 2)
 ' normalize it and also take opsite direction
 nx = nx / nd
 ny = ny / nd
 nz = nz / nd
 
getGrazingAngleInRadian = Application.WorksheetFunction.Pi / 2 - WorksheetFunction.Acos( _
                     (nx * vx + ny * vY + nz * vz) / Sqr(vx ^ 2 + vY ^ 2 + vz ^ 2))
' take only positive
getGrazingAngleInRadian = Abs(getGrazingAngleInRadian)
End Function

Public Function quadrilaterialArea(node1 As ansysCoordinate, node2 As ansysCoordinate, _
                                   node3 As ansysCoordinate, node4 As ansysCoordinate) As Double
    ' Calculate the vectors from point 1 to points 2, 3, and 4
    Dim x1, y1, z1 As Double
    Dim x2, y2, z2 As Double
    Dim x3, y3, z3 As Double
    Dim x4, y4, z4 As Double
    
    x1 = node1.x
    y1 = node1.y
    z1 = node1.z
    
    x2 = node2.x
    y2 = node2.y
    z2 = node2.z
    
    x3 = node3.x
    y3 = node3.y
    z3 = node3.z
    
    x4 = node4.x
    y4 = node4.y
    z4 = node4.z
    
' test only
'    x1 = -1#: y1 = 1#: z1 = 0#
'    x2 = -1#: y2 = -1#: z2 = 0#
'    x3 = 1#: y3 = -1#: z3 = 0#
'    x4 = 1#: y4 = 1#: z4 = 0#
    
    Dim v12(2), v13(2), v14(2) As Double
    v12(0) = x2 - x1
    v12(1) = y2 - y1
    v12(2) = z2 - z1
    v13(0) = x3 - x1
    v13(1) = y3 - y1
    v13(2) = z3 - z1
    v14(0) = x4 - x1
    v14(1) = y4 - y1
    v14(2) = z4 - z1
    
    ' If points 3 and 4 are coincided, use the vector from point 2 to point 4
'    If x3 = x4 And y3 = y4 And z3 = z4 Then
'        v14(0) = x4 - x2
'        v14(1) = y4 - y2
'        v14(2) = z4 - z2
'    End If
    
    ' Calculate the cross products of v12 and v13, and v13 and v14
    Dim c1(2), c2(2) As Double
    c1(0) = v12(1) * v13(2) - v12(2) * v13(1)
    c1(1) = v12(2) * v13(0) - v12(0) * v13(2)
    c1(2) = v12(0) * v13(1) - v12(1) * v13(0)
    c2(0) = v13(1) * v14(2) - v13(2) * v14(1)
    c2(1) = v13(2) * v14(0) - v13(0) * v14(2)
    c2(2) = v13(0) * v14(1) - v13(1) * v14(0)
    
    ' Calculate the magnitudes of the cross products
    Dim mag1, mag2 As Double
    mag1 = Sqr(c1(0) ^ 2 + c1(1) ^ 2 + c1(2) ^ 2)
    mag2 = Sqr(c2(0) ^ 2 + c2(1) ^ 2 + c2(2) ^ 2)
    
    ' Calculate the area of the quadrilateral
    quadrilaterialArea = 0.5 * (mag1 + mag2)
    If quadrilaterialArea = 0 Then
      MsgBox ("this area is zero")
    End If
End Function

Public Function dotProduct(v1 As Variant, v2 As Variant) As Double
   dotProduct = v1(0) * v2(0) + v1(1) * v2(1) + v1(3) * v2(3)
End Function